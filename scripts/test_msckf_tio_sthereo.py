"""
SThereo valley_evening — MSCKF monocular Thermal-Inertial Odometry validation.

Key differences on STheReo:
  - Thermal 14-bit, stereo (we only use the left camera)
  - IMU is an Xsens MTi-300 @ ~200 Hz; noise values in config_sthereo, from the datasheet
  - Extrinsics file is a plain-text 4x4 (in the cam->IMU direction)
  - GT is local_pose.csv, format [t, x, y, z, roll_deg, pitch_deg, yaw_deg]
    (NOT quaternion, degree roll-pitch-yaw)
  - There is a yaw offset between the GT body frame and the MSCKF world
    frame -> rotation is aligned relative to the init yaw
  - Static segment is 3.5 s; init uses the initialize_from_static() method
    (which learns bg, ba and gravity all at once from the Xsens data)

Run:
    python3 scripts/test_msckf_vio_sthereo.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/thermal_vo'))

DL_VER = os.environ.get('TVIO_DL_VER', '2')
if DL_VER == '2':
    from thermal_vo.dataloader2 import ThermalDataLoader, IMULoader, VIOSequencer
else:
    from thermal_vo.dataloader import ThermalDataLoader, IMULoader, VIOSequencer
from thermal_vo.orb        import ORBTracker
from thermal_vo.klt        import KLTTracker
from thermal_vo.msckf      import MSCKF
from thermal_vo.evaluation import (
    align_and_compute_metrics, plot_trajectory, plot_diagnostics,
    append_results_csv, build_csv_row, render_summary_table,
    run_segment_analysis,
)
from thermal_vo            import config_sthereo as config
from thermal_vo.common_params import *   # shared switches, window, ZUPT, PREPROC, MSCKF_PARAMS, make_tracker


# ── FRONT-END ─────────────────────────────────────────────────────────────
METHOD = os.environ.get('TVIO_METHOD', 'orb')        # 'klt' or 'orb'

# ── THERMAL PREPROCESSING ─────────────────────────────────────────────────
# Raw 14-bit thermal; CLAHE's adaptive contrast breaks brightness constancy,
# so it is recommended to keep it off for KLT. It can be turned on for ORB
# if feature scarcity becomes a problem.
USE_CLAHE      = os.environ.get('TVIO_USE_CLAHE', '1') == '1'
GAUSSIAN_SIGMA = PREPROC['gaussian_sigma']

# ── ODOMETRY MODE ─────────────────────────────────────────────────────────
MODE = 'tio'          # 'tio' (full fusion) or 'imu' (pure dead-reckoning)
if MODE not in ('tio', 'imu'):
    raise ValueError(f"Unknown MODE={MODE!r}")

# Shared diagnostic switches (LOCK_ATTITUDE, CHI2_GATE), sliding-window sizes
# (MAX_WINDOW, MAX_TRACKS_PER_UPDATE) and the full ZUPT block come from
# `common_params` via the star import above.

# ── LIVE PLOT ─────────────────────────────────────────────────────────────
LIVE_PLOT = os.environ.get('TVIO_LIVE_PLOT', '0') == '1'
VIZ_EVERY = 2




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

    # STheReo's extrinsics file is correct in 'direct' mode: the file's
    # camera->IMU transform can be interpreted directly as T_imu_cam.
    T_imu_cam     = config.load_extrinsic()
    R_imu_cam_mat = T_imu_cam[:3, :3]
    t_imu_cam     = T_imu_cam[:3, 3]
    R_imu_cam     = R.from_matrix(R_imu_cam_mat)
    print(f"SThereo extrinsic: t_imu_cam={t_imu_cam}  "
          f"|t|={np.linalg.norm(t_imu_cam):.3f} m")

    # ---- DATA ----
    loader = ThermalDataLoader(
        config.THERMAL_LEFT_DIR,
        bit_depth=16,                # 14-bit raw stored as 16-bit PNG
        undistort=True, K=K, D=D,
        use_clahe=USE_CLAHE,
        **PREPROC,
    )
    imu = IMULoader.sthereo(config.IMU_CSV)
    print(f"images: {len(loader)}    IMU: {len(imu)}")

    # ---- MSCKF ----
    msckf = MSCKF(
        K=K, D=None,                # undistortion zaten loader'da
        **MSCKF_PARAMS,             # shared back-end config (common_params)
    )

    # Process noise covariance — config_sthereo Xsens MTi-300 datasheet
    
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

    # ---- INIT (initialize_from_static — STheReo's working init method) ----
    # bg, ba and gravity are estimated from the 3.5 s static IMU window.
    n_static = int(config.STATIC_INIT_SECONDS / np.median(np.diff(imu.timestamps)))
    msckf.initialize_from_static(imu.gyro[:n_static], imu.accel[:n_static])
    msckf.lock_imu_attitude = LOCK_ATTITUDE
    msckf.chi2_enabled      = CHI2_GATE

    t_init = imu.timestamps[n_static - 1]
    msckf_yaw_init_deg = msckf.nominal_rot.as_euler('xyz', degrees=True)[2]
    print(f"static init: {n_static} samples, |g|={np.linalg.norm(msckf.gravity):.4f}")
    print(f"  bg(init)={msckf.bg}")
    print(f"  ba(init)={msckf.ba}")
    print(f"  MSCKF yaw at init: {msckf_yaw_init_deg:+.2f}°")

    # ---- GT (local_pose.csv: t, x, y, z, roll_deg, pitch_deg, yaw_deg) ----
    gt     = np.loadtxt(config.GT_LOCAL_POSE, delimiter=',')
    gt_t   = gt[:, 0]
    gt_xyz = gt[:, 1:4]
    gt_rpy = gt[:, 4:7]
    print(f"GT samples: {len(gt_t)}, t = [{gt_t[0]:.3f}, {gt_t[-1]:.3f}] s")

    # Align GT's yaw with MSCKF's yaw at the init instant — a rotation-only frame transform.
    gt_yaw_init_deg = float(np.interp(t_init, gt_t, gt_rpy[:, 2]))
    R_gt_to_msckf = R.from_euler(
        'z', msckf_yaw_init_deg - gt_yaw_init_deg, degrees=True
    ).as_matrix()
    gt_xyz_aligned = gt_xyz @ R_gt_to_msckf.T

    # GT origin: shift the position at the init instant to MSCKF's
    # (msckf.nominal_pos) origin, so both trajectories start at the same point.
    gt_pos_init = np.array([
        np.interp(t_init, gt_t, gt_xyz_aligned[:, i]) for i in range(3)
    ])
    gt_offset   = msckf.nominal_pos - gt_pos_init
    gt_xyz_aligned = gt_xyz_aligned + gt_offset
    print(f"GT yaw at init: {gt_yaw_init_deg:+.2f}°  "
          f"→ aligned to MSCKF yaw  (offset {msckf_yaw_init_deg - gt_yaw_init_deg:+.2f}°)")

    # ---- TRACKER ----
    tracker = make_tracker(METHOD)
    print(f"front-end: {METHOD.upper()}  mode={MODE}  CHI2={CHI2_GATE}  "
          f"LOCK_ATT={LOCK_ATTITUDE}  max_tracks/upd={MAX_TRACKS_PER_UPDATE}  "
          f"CLAHE={USE_CLAHE}")

    # ---- LIVE PLOT ----
    gt_x, gt_y, gt_z = gt_xyz_aligned[:, 0], gt_xyz_aligned[:, 1], gt_xyz_aligned[:, 2]
    if LIVE_PLOT:
        plt.ion()
        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
        ax_xy  = fig.add_subplot(gs[0, 0])
        ax_z   = fig.add_subplot(gs[0, 1])
        ax_img = fig.add_subplot(gs[1, :])
        gt_t_rel = gt_t - t_init

        gt_xy_line,  = ax_xy.plot([], [], 'r--', lw=1.3, alpha=0.85, label='GT (local pose)')
        tio_xy_line, = ax_xy.plot([], [], 'b-',  lw=1.3, label='TIO')
        gt_xy_pt     = ax_xy.scatter([], [], c='r', s=70, zorder=5, edgecolor='k')
        tio_xy_pt    = ax_xy.scatter([], [], c='b', s=70, zorder=5, edgecolor='k')
        ax_xy.scatter([0.0], [0.0], c='g', s=80, marker='o', zorder=5, label='start')
        ax_xy.set_xlabel('x [m]'); ax_xy.set_ylabel('y [m]')
        ax_xy.set_aspect('equal'); ax_xy.grid(alpha=0.3); ax_xy.legend(loc='best')
        ax_xy.set_title('Top-down (XY) — bounds auto-scale')

        gt_z_line,  = ax_z.plot([], [], 'r--', lw=1.3, alpha=0.85, label='GT (local pose)')
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

        fig.suptitle(f"SThereo valley_evening — MSCKF TIO ({METHOD.upper()})  |  t =   0.0 s",
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
    track_log      = []   # (t, n_active, n_used) per cam frame

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

        # ── LIVE PLOT REDRAW ───────────────────────────────────────────────
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
            gt_mask = gt_t <= t
            gt_disp_x = gt_x[gt_mask] - gt_orig[0]
            gt_disp_y = gt_y[gt_mask] - gt_orig[1]
            gt_disp_z = gt_z[gt_mask] - gt_orig[2]
            gt_xy_line.set_data(gt_disp_x, gt_disp_y)
            gt_z_line.set_data(gt_t_rel[gt_mask], gt_disp_z)
            gx_now = float(np.interp(t, gt_t, gt_x)) - gt_orig[0]
            gy_now = float(np.interp(t, gt_t, gt_y)) - gt_orig[1]
            gz_now = float(np.interp(t, gt_t, gt_z)) - gt_orig[2]
            gt_xy_pt.set_offsets([[gx_now, gy_now]])
            gt_z_pt.set_offsets([[t - t_init, gz_now]])

            # Dynamic bounds: union of (GT shown so far) and (TIO so far) + pad.
            xs = np.concatenate([gt_disp_x, vp_disp[:, 0]])
            ys = np.concatenate([gt_disp_y, vp_disp[:, 1]])
            zs = np.concatenate([gt_disp_z, vp_disp[:, 2]])
            pad_xy = max(1.0, 0.1 * max(xs.ptp(), ys.ptp()))
            pad_z  = max(0.3, 0.1 * zs.ptp())
            ax_xy.set_xlim(xs.min() - pad_xy, xs.max() + pad_xy)
            ax_xy.set_ylim(ys.min() - pad_xy, ys.max() + pad_xy)
            ax_z.set_ylim(zs.min() - pad_z, zs.max() + pad_z)

            # Two-frame overlay: prev | curr stacked horizontally.
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
            prev_pts.set_offsets(np.asarray(prev_xy) if prev_xy
                                 else np.empty((0, 2)))
            curr_pts.set_offsets(np.asarray(curr_xy) if curr_xy
                                 else np.empty((0, 2)))

            n_active = len(tracker.active_tracks) if MODE == 'tio' else 0
            fig.suptitle(
                f"SThereo valley_evening — MSCKF TIO ({METHOD.upper()})  |  "
                f"t = {t - t_init:5.1f} s   active tracks: {n_active}",
                fontsize=12,
            )
            plt.pause(0.001)

        prev_img_cache = img

        # GT comparison log (first 40 + every 25th)
        if n_frame <= 40 or n_frame % 25 == 0:
            i = int(np.searchsorted(gt_t, t))
            if 0 < i < len(gt_t):
                gp = np.array([gt_x[i], gt_y[i], gt_z[i]])
                eu_m = msckf.nominal_rot.as_euler('xyz', degrees=True)
                gr = float(np.interp(t, gt_t, gt_rpy[:, 0]))
                gp_pitch = float(np.interp(t, gt_t, gt_rpy[:, 1]))
                gy = float(np.interp(t, gt_t, gt_rpy[:, 2]))
                err_pos   = float(np.linalg.norm(msckf.nominal_pos - gp))
                roll_err  = wrap_deg(eu_m[0] - gr)
                pitch_err = wrap_deg(eu_m[1] - gp_pitch)
                yaw_err   = wrap_deg(wrap_deg(eu_m[2] - msckf_yaw_init_deg)
                                     - wrap_deg(gy - gt_yaw_init_deg))
                print(f"  [att] f{n_frame:4d} t={t-t_init:6.1f}s  "
                      f"err_pos={err_pos:7.3f}m  "
                      f"roll_err={roll_err:+6.2f}  pitch_err={pitch_err:+6.2f}  "
                      f"yaw_err={yaw_err:+6.2f}  "
                      f"|vel|={np.linalg.norm(msckf.nominal_vel):6.2f}m/s",
                      flush=True)

    # ---- SAVE + FINAL STATS ----
    traj = np.array(traj)
    print(f"\nmode={MODE}   processed: {n_predict} IMU predicts, {n_frame} cam frames,"
          f"  ZUPTs fired: {n_zupt}")

    if len(traj):
        metrics = align_and_compute_metrics(traj, gt_t, gt_xyz_aligned)
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

        print(f"\nvs SThereo local-pose reference:")
        print(f"  ATE RMSE      : {metrics['ate_rmse_m']:.3f} m")
        print(f"  ATE max       : {metrics['ate_max_m']:.3f} m")
        print(f"  drift rate    : {metrics['drift_pct']:.2f} %")
        print(f"  survival      : {metrics['survival_s']:.1f} s (threshold 5 m)")
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
        out_base = (f"sthereo_valley_{METHOD.upper()}_CLAHE{'ON' if USE_CLAHE else 'OFF'}_{norm_order}"
                    + ('' if MODE == 'tio' else f"_{MODE.upper()}"))
        out_txt = f"results/{out_base}.txt"
        out_png_traj = f"results/{out_base}_traj.png"
        out_png_diag = f"results/{out_base}_diag.png"
        out_csv = "results/all_runs.csv"
        os.makedirs('results', exist_ok=True)
        np.savetxt(out_txt, traj, header='t x y z qx qy qz qw')

        title_info = {
            'run_id':         out_base,
            'dataset':        'sthereo_valley',
            'method':         METHOD,
            'mode':           MODE,
            'clahe':          USE_CLAHE,
            'gaussian_sigma': GAUSSIAN_SIGMA,
            'normalize':      'percentile',
            'chi2_alpha':     msckf.chi2_alpha,
            'max_window':     MAX_WINDOW,
        }
        run_segment_analysis(
            out_base, traj, metrics, gt_t, gt_xyz_aligned,
            config.SEG_THRESHOLD_M, config.SEG_MIN_DURATION_S,
            track_series=track_series, nis_log=msckf.nis_log)
        plot_trajectory(traj, gt_t, gt_xyz_aligned, metrics,
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
