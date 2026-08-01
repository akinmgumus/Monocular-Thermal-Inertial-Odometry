"""
ROVTIO alt1 — MSCKF monocular Thermal-Inertial Odometry validation.

Key differences on ROVTIO alt1 (NTNU ARL):
  - Raw 16-bit thermal (FLIR Tau2, 640x512, ~25 Hz)
  - VectorNav VN-100 IMU @ 168 Hz, equidistant (fisheye) thermal lens
  - Vicon motion-capture GT (true ground truth, ~6 Hz)
  - Indoor scene, drone is STATIC at the start -> gravity-aligned init works
  - IMU-cam timeshift: -24.14 ms (the cam clock leads the cam; add +24.14 ms to the IMU)
  - Calibration: from the cfg/rovtio files (charlie_tau2.yaml, rovtio.info)

Init strategy (same as FIReStereo's gravity-aligned init):
  - pos        : Vicon GT[0]
  - roll/pitch : from the accel mean over the first N seconds (drone genuinely static)
  - yaw        : from Vicon GT[0]'s yaw
  - vel        : 0 (drone static)
  - bg         : gyro mean over the first static segment
  - ba         : a_mean - R_init^T @ g
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/mTIO'))

DL_VER = os.environ.get('TVIO_DL_VER', '2') 
# 1 = CLAHE-NORM
# 2 = NORM-CLAHE

# ── FRONT-END ─────────────────────────────────────────────────────────────
METHOD = os.environ.get('TVIO_METHOD', 'orb')
# ── THERMAL PREPROCESSING ─────────────────────────────────────────────────
# Raw 16-bit FLIR Tau2 — CLAHE opens up the thermal contrast, helpful here.
# The indoor scene is usually in a narrow dynamic range, so percentile
# normalisation is useful.
USE_CLAHE      = os.environ.get('TVIO_USE_CLAHE', '1') == '1'


if DL_VER == '2':
    from mTIO.dataloader2 import ThermalDataLoader, IMULoader, VIOSequencer
else:
    from mTIO.dataloader import ThermalDataLoader, IMULoader, VIOSequencer
from mTIO.orb        import ORBTracker
from mTIO.klt        import KLTTracker
from mTIO.msckf      import MSCKF
from mTIO.evaluation import (
    align_and_compute_metrics, plot_trajectory, plot_diagnostics,
    append_results_csv, build_csv_row, render_summary_table,
    run_segment_analysis,
)
from mTIO            import config_rovtio as config
from mTIO.common_params import *   # shared switches, window, ZUPT, PREPROC, MSCKF_PARAMS, make_tracker


GAUSSIAN_SIGMA = PREPROC['gaussian_sigma']



# ── ODOMETRY MODE ─────────────────────────────────────────────────────────
MODE = 'tio'      # 'tio' (full fusion) or 'imu' (pure dead-reckoning)
if MODE not in ('tio', 'imu'):
    raise ValueError(f"Unknown MODE={MODE!r}")

# Shared diagnostic switches (LOCK_ATTITUDE, CHI2_GATE), the sliding-window
# sizes (MAX_WINDOW, MAX_TRACKS_PER_UPDATE) and the full ZUPT block come from
# `common_params` via the star import above.

# ── LIVE PLOT ─────────────────────────────────────────────────────────────
LIVE_PLOT = os.environ.get('TVIO_LIVE_PLOT', '0') == '1'
VIZ_EVERY = 1

# ── STATIC INIT WINDOW ────────────────────────────────────────────────────
# The drone is static at the start — the 0.5..3.0 s window should be clean
# enough for the gravity-aligned init. Yaw is taken from the Vicon GT.
INIT_STATIC_T0 = 0.5
INIT_STATIC_T1 = 12.0


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def is_stationary(imu_buffer, gyro_thresh, accel_dev_thresh, g=9.81):
    if len(imu_buffer) < 10:
        return False
    arr = np.asarray(imu_buffer)
    gyro_norm  = np.linalg.norm(arr[:, 1:4], axis=1).mean()
    accel_norm = np.linalg.norm(arr[:, 4:7], axis=1).mean()
    return (gyro_norm < gyro_thresh and abs(accel_norm - g) < accel_dev_thresh)


def run():
    # ---- CALIBRATION ----
    cal = config.load_camera_intrinsics()
    K, D = cal['K'], cal['D']
    dist_model = cal['distortion_model']

    T_imu_cam     = config.load_extrinsic_imu_cam()
    R_imu_cam_mat = T_imu_cam[:3, :3]
    t_imu_cam     = T_imu_cam[:3, 3]
    R_imu_cam     = R.from_matrix(R_imu_cam_mat)
    print(f"ROVTIO extrinsic: t_imu_cam={t_imu_cam}  "
          f"|t|={np.linalg.norm(t_imu_cam):.3f} m")

    # IMU clock → cam clock alignment.
    # timeshift = t_imu - t_cam = -0.0241 → IMU is BEHIND cam by 24 ms.
    # IMULoader t_offset is added to every IMU timestamp.
    timeshift = config.load_timeshift_imu_cam()
    t_offset  = -timeshift
    print(f"timeshift_imu_cam = {timeshift*1000:+.2f} ms  →  IMU t_offset = {t_offset*1000:+.2f} ms")

    # ---- DATA ----
    loader = ThermalDataLoader(
        config.CAM0_DIR,
        bit_depth=16,
        undistort=True, K=K, D=D,
        distortion_model=dist_model,
        use_clahe=USE_CLAHE,
        **PREPROC,
    )

    loader.load_timestamps_file(config.CAM0_TS, unit='s')

    imu = IMULoader(config.IMU_CSV, t_offset=t_offset)
    gt  = config.load_ground_truth()
    print(f"images: {len(loader)}    IMU: {len(imu)}    "
          f"cam dur: {loader.timestamps[-1]-loader.timestamps[0]:.1f} s")
    print(f"GT (Vicon): {len(gt['t'])} samples, {gt['t'][-1]-gt['t'][0]:.1f} s")

    # ---- MSCKF ----
    msckf = MSCKF(
        K=K, D=None,                # undistortion zaten loader'da
        **MSCKF_PARAMS,             # shared params (common_params)
    )

    vg  = config.IMU_GYRO_NOISE_DENSITY  ** 2
    va  = config.IMU_ACCEL_NOISE_DENSITY ** 2
    vbg = config.IMU_GYRO_BIAS_RW        ** 2
    vba = config.IMU_ACCEL_BIAS_RW       ** 2
    msckf.Q_matrix = np.diag([
        vg, vg, vg,
        vbg, vbg, vbg,
        va, va, va,
        vba, vba, vba,
        0.0, 0.0, 0.0,
    ])

    # ---- INIT (gravity-aligned, statik segmentten) ----
    t_init = gt['t'][0]
    msckf.nominal_pos = gt['p'][0].copy()
    msckf.nominal_vel = np.zeros(3)
    msckf.gravity     = np.array([0.0, 0.0, config.GRAVITY_MAGNITUDE])

    # First static segment IMU statistics
    t0 = t_init + INIT_STATIC_T0
    t1 = t_init + INIT_STATIC_T1
    mask = (imu.timestamps >= t0) & (imu.timestamps <= t1)
    if mask.sum() < 100:
        sys.exit(f"Not enough IMU samples in the init window ({mask.sum()}); "
                 "adjust INIT_STATIC_T0/T1.")
    g_init = imu.gyro[mask]
    a_init = imu.accel[mask]
    gyro_norm_init = np.linalg.norm(g_init, axis=1).mean()
    a_mean         = a_init.mean(axis=0)
    a_norm         = np.linalg.norm(a_mean)
    print(f"\nInit window [{INIT_STATIC_T0}, {INIT_STATIC_T1}] s: "
          f"{mask.sum()} sample, |w|={gyro_norm_init:.4f} rad/s, |a|={a_norm:.3f}")

    # Yaw: from Vicon GT[0]
    qx, qy, qz, qw = gt['q'][0]
    yaw_gt = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=True)[2]

    # Roll/pitch: gravity-aligned, from the accel mean
    roll_acc  = float(np.degrees(np.arctan2( a_mean[1], a_mean[2])))
    pitch_acc = float(np.degrees(np.arctan2(-a_mean[0],
                                              np.sqrt(a_mean[1]**2 + a_mean[2]**2))))
    R_init = R.from_euler('xyz', [roll_acc, pitch_acc, yaw_gt], degrees=True)
    msckf.nominal_rot     = R_init
    msckf.nominal_rot_fej = R_init

    # bg: mean gyro over the first static segment (genuinely static -> a reliable bias estimate)
    msckf.bg = g_init.mean(axis=0)

    # ba: should be small, since we are gravity-aligned
    g_body_expected = R_init.as_matrix().T @ msckf.gravity
    msckf.ba = a_mean - g_body_expected

    msckf.lock_imu_attitude = LOCK_ATTITUDE
    msckf.chi2_enabled      = CHI2_GATE

    init_rpy = msckf.nominal_rot.as_euler('xyz', degrees=True)
    print(f"\ngravity-init @ t={t_init:.3f}s  pos={msckf.nominal_pos}")
    print(f"  attitude  : roll={init_rpy[0]:+.2f}°  pitch={init_rpy[1]:+.2f}°  "
          f"yaw={init_rpy[2]:+.2f}° (yaw from Vicon)")
    print(f"  bg(init)  : {msckf.bg}")
    print(f"  ba(init)  : {msckf.ba}    |ba|={np.linalg.norm(msckf.ba):.4f}")

    # ---- TRACKER ----
    tracker = make_tracker(METHOD)
    print(f"\nfront-end: {METHOD.upper()}  mode={MODE}  CHI2={CHI2_GATE}  "
          f"LOCK_ATT={LOCK_ATTITUDE}  max_tracks/upd={MAX_TRACKS_PER_UPDATE}  "
          f"CLAHE={USE_CLAHE}")

    # ---- LIVE PLOT ----
    gt_x, gt_y, gt_z = gt['p'][:, 0], gt['p'][:, 1], gt['p'][:, 2]
    if LIVE_PLOT:
        plt.ion()
        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
        ax_xy  = fig.add_subplot(gs[0, 0])
        ax_z   = fig.add_subplot(gs[0, 1])
        ax_img = fig.add_subplot(gs[1, :])

        gt_t_rel = gt['t'] - t_init

        gt_xy_line,  = ax_xy.plot([], [], 'r--', lw=1.3, alpha=0.85, label='Vicon GT')
        tio_xy_line, = ax_xy.plot([], [], 'b-',  lw=1.3, label='TIO')
        gt_xy_pt     = ax_xy.scatter([], [], c='r', s=70, zorder=5, edgecolor='k')
        tio_xy_pt    = ax_xy.scatter([], [], c='b', s=70, zorder=5, edgecolor='k')
        ax_xy.scatter([0.0], [0.0], c='g', s=80, marker='o', zorder=5, label='start')
        ax_xy.set_xlabel('x [m]'); ax_xy.set_ylabel('y [m]')
        ax_xy.set_aspect('equal'); ax_xy.grid(alpha=0.3); ax_xy.legend(loc='best')
        ax_xy.set_title('Top-down (XY)')

        gt_z_line,  = ax_z.plot([], [], 'r--', lw=1.3, alpha=0.85, label='Vicon GT')
        tio_z_line, = ax_z.plot([], [], 'b-',  lw=1.3, label='TIO')
        gt_z_pt     = ax_z.scatter([], [], c='r', s=70, zorder=5, edgecolor='k')
        tio_z_pt    = ax_z.scatter([], [], c='b', s=70, zorder=5, edgecolor='k')
        ax_z.set_xlim(gt_t_rel.min(), gt_t_rel.max())
        ax_z.set_xlabel('t [s]'); ax_z.set_ylabel('z [m]')
        ax_z.grid(alpha=0.3); ax_z.legend(loc='best')
        ax_z.set_title('Height (Z) over time')

        first_img, _ = loader[0]
        IMG_W = first_img.shape[1]
        img_artist = ax_img.imshow(np.hstack([first_img, first_img]),
                                   cmap='gray', vmin=0, vmax=255)
        ax_img.set_title('Left: frame N-1 | Right: frame N  '
                         '(blue=prev pt, green=curr pt, yellow=match)')
        ax_img.set_xticks([]); ax_img.set_yticks([])
        ax_img.axvline(IMG_W - 0.5, color='white', lw=0.8, alpha=0.6)
        match_lines, = ax_img.plot([], [], color='yellow', lw=0.5, alpha=0.55)
        prev_pts = ax_img.scatter([], [], s=14, c='deepskyblue', edgecolor='black',
                                  linewidths=0.4, zorder=5)
        curr_pts = ax_img.scatter([], [], s=14, c='lime', edgecolor='black',
                                  linewidths=0.4, zorder=5)

        fig.suptitle(f"ROVTIO alt1 — MSCKF TIO ({METHOD.upper()})  |  t =   0.0 s",
                     fontsize=12)

    # ---- EVENT LOOP ----
    seq = VIOSequencer(loader, imu)
    traj       = []
    prev_imu_t = t_init
    frame_id   = 0
    n_predict  = 0
    n_frame    = 0
    n_zupt     = 0
    imu_buffer = []
    zupt_ctr   = 0
    prev_img_cache = None
    track_log  = []   # (t, n_active, n_used) per cam frame, for 4-panel plot

    for kind, t, payload in seq.events():
        if t <= t_init:
            continue

        if kind == 'imu':
            dt = t - prev_imu_t
            if dt > 0:
                msckf.predict(dt, payload[1:4], payload[4:7])
            prev_imu_t = t
            n_predict += 1

            if ZUPT:
                imu_buffer.append(payload)
                if len(imu_buffer) > ZUPT_IMU_WINDOW:
                    imu_buffer.pop(0)
                zupt_ctr += 1
                if zupt_ctr % ZUPT_THROTTLE == 0:
                    if is_stationary(imu_buffer, ZUPT_GYRO_THRESH, ZUPT_ACCEL_DEV_THRESH):
                        msckf.zero_velocity_update(sigma_zupt=ZUPT_SIGMA)
                        n_zupt += 1
            continue

        # kind == 'cam'
        img = payload

        if MODE == 'tio':
            msckf.now = t                    # NIS log zaman damgasi
            msckf.augment_state(frame_id, R_imu_cam, t_imu_cam)
            tracker.process_frame(img, frame_id)

            n_drop = len(msckf.cam_states) - MAX_WINDOW
            if n_drop > 0:
                for cs in msckf.cam_states[:n_drop]:
                    tracker.marginalize_at_prune(cs.frame_id)

            upd_tracks = list(tracker.dead_tracks)
            if len(upd_tracks) > MAX_TRACKS_PER_UPDATE:
                upd_tracks.sort(key=lambda tr: len(tr.frame_ids), reverse=True)
                upd_tracks = upd_tracks[:MAX_TRACKS_PER_UPDATE]

            msckf.update(upd_tracks)
            msckf.prune_cam_states(MAX_WINDOW)

        traj.append((t, *msckf.nominal_pos, *msckf.nominal_rot.as_quat()))
        n_active = len(tracker.active_tracks) if MODE == 'tio' else 0
        track_log.append((t, n_active,
                          msckf.last_n_used if MODE == 'tio' else 0))
        frame_id += 1
        n_frame  += 1

        if LIVE_PLOT and n_frame % VIZ_EVERY == 0 and len(traj) >= 2:
            traj_arr = np.array(traj)
            vt_rel   = traj_arr[:, 0] - t_init
            vp_arr   = traj_arr[:, 1:4]
            vp_orig  = vp_arr[0].copy()
            gt_orig  = np.array([gt_x[0], gt_y[0], gt_z[0]])
            vp_disp  = vp_arr - vp_orig
            tio_xy_line.set_data(vp_disp[:, 0], vp_disp[:, 1])
            tio_xy_pt.set_offsets([[vp_disp[-1, 0], vp_disp[-1, 1]]])
            tio_z_line.set_data(vt_rel, vp_disp[:, 2])
            tio_z_pt.set_offsets([[vt_rel[-1], vp_disp[-1, 2]]])
            gt_mask = gt['t'] <= t
            gt_disp_x = gt_x[gt_mask] - gt_orig[0]
            gt_disp_y = gt_y[gt_mask] - gt_orig[1]
            gt_disp_z = gt_z[gt_mask] - gt_orig[2]
            gt_xy_line.set_data(gt_disp_x, gt_disp_y)
            gt_z_line.set_data(gt_t_rel[gt_mask], gt_disp_z)
            gx_now = float(np.interp(t, gt['t'], gt_x)) - gt_orig[0]
            gy_now = float(np.interp(t, gt['t'], gt_y)) - gt_orig[1]
            gz_now = float(np.interp(t, gt['t'], gt_z)) - gt_orig[2]
            gt_xy_pt.set_offsets([[gx_now, gy_now]])
            gt_z_pt.set_offsets([[t - t_init, gz_now]])

            xs = np.concatenate([gt_disp_x, vp_disp[:, 0]])
            ys = np.concatenate([gt_disp_y, vp_disp[:, 1]])
            zs = np.concatenate([gt_disp_z, vp_disp[:, 2]])
            pad_xy = max(0.3, 0.1 * max(xs.ptp(), ys.ptp()))
            pad_z  = max(0.1, 0.1 * zs.ptp())
            ax_xy.set_xlim(xs.min() - pad_xy, xs.max() + pad_xy)
            ax_xy.set_ylim(ys.min() - pad_xy, ys.max() + pad_xy)
            ax_z.set_ylim(zs.min() - pad_z, zs.max() + pad_z)

            disp_prev = prev_img_cache if prev_img_cache is not None else img
            img_artist.set_data(np.hstack([disp_prev, img]))
            seg_x, seg_y = [], []
            prev_xy, curr_xy = [], []
            if MODE == 'tio':
                for tr in tracker.active_tracks:
                    if len(tr.keypoints) >= 2:
                        p0 = tr.keypoints[-2]
                        p1 = tr.keypoints[-1]
                        prev_xy.append(p0)
                        curr_xy.append((p1[0] + IMG_W, p1[1]))
                        seg_x += [p0[0], p1[0] + IMG_W, np.nan]
                        seg_y += [p0[1], p1[1], np.nan]
            match_lines.set_data(seg_x, seg_y)
            prev_pts.set_offsets(np.asarray(prev_xy) if prev_xy else np.empty((0, 2)))
            curr_pts.set_offsets(np.asarray(curr_xy) if curr_xy else np.empty((0, 2)))

            n_active = len(tracker.active_tracks) if MODE == 'tio' else 0
            fig.suptitle(
                f"ROVTIO alt1 — MSCKF TIO ({METHOD.upper()})  |  "
                f"t = {t - t_init:5.1f} s   active tracks: {n_active}",
                fontsize=12,
            )
            plt.pause(0.001)

        prev_img_cache = img

        if n_frame <= 40 or n_frame % 25 == 0:
            i = int(np.searchsorted(gt['t'], t))
            if 0 < i < len(gt['t']):
                gp = gt['p'][i]
                gq = gt['q'][i]
                gt_R = R.from_quat([gq[0], gq[1], gq[2], gq[3]])
                eu_m = msckf.nominal_rot.as_euler('xyz', degrees=True)
                eu_g = gt_R.as_euler('xyz', degrees=True)
                err_pos = float(np.linalg.norm(msckf.nominal_pos - gp))
                print(f"  [att] f{n_frame:4d} t={t-t_init:6.1f}s  "
                      f"err_pos={err_pos:7.3f}m  "
                      f"roll_err={wrap_deg(eu_m[0]-eu_g[0]):+6.2f}  "
                      f"pitch_err={wrap_deg(eu_m[1]-eu_g[1]):+6.2f}  "
                      f"yaw_err={wrap_deg(eu_m[2]-eu_g[2]):+6.2f}  "
                      f"|vel|={np.linalg.norm(msckf.nominal_vel):6.2f}m/s",
                      flush=True)

    # ---- SAVE + FINAL STATS ----
    traj = np.array(traj)
    print(f"\nmode={MODE}   processed: {n_predict} IMU predicts, {n_frame} cam frames,"
          f"  ZUPTs: {n_zupt}")

    if len(traj):
        metrics = align_and_compute_metrics(traj, gt['t'], gt['p'])
        metrics['nis_pass_rate'] = (msckf.cum_n_used / msckf.cum_n_in
                                    if msckf.cum_n_in > 0 else None)
        metrics['mean_nis'] = msckf.mean_nis   # avg NIS/df, ~1 is consistent

        track_arr = np.asarray(track_log) if track_log else None
        track_series = None
        if track_arr is not None and len(track_arr):
            track_series = {
                't':      track_arr[:, 0],
                'active': track_arr[:, 1],
                'used':   track_arr[:, 2],
            }
            metrics['avg_active_tracks']   = float(track_arr[:, 1].mean())
            metrics['avg_used_per_update'] = float(track_arr[:, 2].mean())

        print(f"\nvs Vicon GT (real motion capture):")
        print(f"  ATE RMSE      : {metrics['ate_rmse_m']:.3f} m")
        print(f"  ATE max       : {metrics['ate_max_m']:.3f} m")
        print(f"  drift rate    : {metrics['drift_pct']:.2f} %")
        print(f"  survival      : {metrics['survival_s']:.1f} s (threshold 2 m)")
        print(f"  RPE 1s trans  : {metrics['rpe_trans_rmse']:.3f} m")
        if metrics['nis_pass_rate'] is not None:
            print(f"  used/in       : {metrics['nis_pass_rate']:.2f}  "
                  f"({msckf.cum_n_used}/{msckf.cum_n_in} over {msckf.n_updates} updates)")
        if metrics.get('mean_nis') is not None:
            print(f"  mean NIS/df   : {metrics['mean_nis']:.2f}  (~1 tutarli; >>1 overconfident)")
        if metrics.get('diverged'):
            print(f"  divergence    : @ {metrics['divergence_time_s']:.1f} s  "
                  f"(pre-div ATE RMSE = {metrics['pre_divergence_ate_rmse_m']:.3f} m)")
        else:
            print(f"  divergence    : none (survived full {metrics['duration_s']:.1f} s)")

        norm_order = "NORM-CLAHE" if "dataloader2" in ThermalDataLoader.__module__ else "CLAHE-NORM"
        out_base = (f"rovtio_alt1_{METHOD.upper()}_CLAHE{'ON' if USE_CLAHE else 'OFF'}_{norm_order}"
                    + ('' if MODE == 'tio' else f"_{MODE.upper()}"))
        out_txt = f"results/{out_base}.txt"
        out_png_traj = f"results/{out_base}_traj.png"
        out_png_diag = f"results/{out_base}_diag.png"
        out_csv = "results/all_runs.csv"
        os.makedirs('results', exist_ok=True)
        np.savetxt(out_txt, traj, header='t x y z qx qy qz qw')

        title_info = {
            'run_id':         out_base,
            'dataset':        'rovtio_alt1',
            'method':         METHOD,
            'mode':           MODE,
            'clahe':          USE_CLAHE,
            'gaussian_sigma': GAUSSIAN_SIGMA,
            'normalize':      'percentile',
            'chi2_alpha':     msckf.chi2_alpha,
            'max_window':     MAX_WINDOW,
        }
        run_segment_analysis(
            out_base, traj, metrics, gt['t'], gt['p'],
            config.SEG_THRESHOLD_M, config.SEG_MIN_DURATION_S,
            track_series=track_series, nis_log=msckf.nis_log)
        plot_trajectory(traj, gt['t'], gt['p'], metrics,
                        title_info=title_info, out_png=out_png_traj)
        if track_series is not None:
            np.savez(f"results/{out_base}_tracks.npz", **track_series)
        plot_diagnostics(traj, metrics, track_series,
                         title_info=title_info, out_png=out_png_diag)

        append_results_csv(out_csv, build_csv_row(title_info, metrics))
        render_summary_table(out_csv, 'results/summary_table.png')
        print(f"\nsaved → {out_txt}")
        print(f"saved → {out_png_traj}")
        print(f"saved → {out_png_diag}")
        print(f"appended → {out_csv}")
        print(f"updated → results/summary_table.png")

    if LIVE_PLOT:
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    run()