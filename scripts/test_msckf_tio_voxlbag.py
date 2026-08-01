"""
VOXL Starling 2 Max — MSCKF monocular Thermal-Inertial Odometry validation.

Key differences on the VOXL bag (decoded_first_2min_v3):
  - Thermal mono8 (H.264-decoded; the original 16-bit FLIR Boson+ information is lost).
  - IMU is /imu_apps @ 472 Hz; noise/bias parameters in config_voxlbag are a
    PLACEHOLDER (VINS-style conservative tuning from drone_config.yaml).
  - IMU-cam time offset +15 ms.
  - There is NO static segment at all in this 2-minute stretch — the drone is
    manoeuvring from the very start. R_init comes from PX4 odom's first pose;
    ba_init comes from the quasi-static window [2.5, 6.0] s.
  - GT: two references. /px4/odom (EKF2, drifts but is accurate) and
    /fmu/.../gps (independent but noisy, ~1.8 m noise). ATE is reported
    against both.
  - The drone is fast: max ~7 m/s, 820 m E-W, 95 m of climb. Strong parallax
    — much better conditions for monocular than FIReStereo.

Run:
    python3 scripts/test_msckf_vio_voxlbag.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/mTIO'))

DL_VER = os.environ.get('TVIO_DL_VER', '1')
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
from mTIO            import config_voxlbag as config
from mTIO.common_params import *   # shared switches, window, ZUPT, PREPROC, MSCKF_PARAMS, make_tracker


# ── FRONT-END ─────────────────────────────────────────────────────────────
METHOD = os.environ.get('TVIO_METHOD', 'klt')   # 'klt' or 'orb'

# ── THERMAL PREPROCESSING ─────────────────────────────────────────────────
# H264-decoded mono8 thermal — CLAHE inflates compression artefacts, keep it
# off. Gaussian blur gives a light denoise.
USE_CLAHE      = os.environ.get('TVIO_USE_CLAHE', '1') == '1'
GAUSSIAN_SIGMA = PREPROC['gaussian_sigma']

# ── ODOMETRY MODE ─────────────────────────────────────────────────────────
MODE = 'tio'      # 'tio' (full fusion) or 'imu' (pure dead-reckoning)
if MODE not in ('tio', 'imu'):
    raise ValueError(f"Unknown MODE={MODE!r}")

# ── GYRO SIGN SWEEP ────────────────────────────────────────────────────────
# The /imu_apps frame convention is documented as ambiguous by the dataset
# owner. To find the right gyro sign convention empirically, set this to one
# of the 8 sign tuples and re-run with MODE='imu'. The combination that makes
# the trajectory direction and amplitude match GT for the first ~5 s wins.
#   ( 1,  1,  1) raw
#   (-1,  1,  1) negate X only       ← partially-working result you saw
#   ( 1, -1,  1) negate Y only
#   ( 1,  1, -1) negate Z only
#   (-1, -1,  1) negate X, Y
#   (-1,  1, -1) negate X, Z
#   ( 1, -1, -1) negate Y, Z         ← FRD→FLU flip hypothesis
#   (-1, -1, -1) negate all
GYRO_SIGN = (1, 1, 1)   # change this and re-run; default = FRD→FLU flip

# Shared diagnostic switches (LOCK_ATTITUDE, CHI2_GATE), sliding-window sizes
# (MAX_WINDOW, MAX_TRACKS_PER_UPDATE) and the full ZUPT block come from
# `common_params` via the star import above.

# ── LIVE PLOT ─────────────────────────────────────────────────────────────
LIVE_PLOT = os.environ.get('TVIO_LIVE_PLOT', '0') == '1'
VIZ_EVERY = 2




def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def run():
    # ---- CALIBRATION ----
    cal = config.load_camera_intrinsics()
    K, D = cal['K'], cal['D']

    T_imu_cam     = config.load_extrinsic_imu_cam()
    R_imu_cam_mat = T_imu_cam[:3, :3]
    t_imu_cam     = T_imu_cam[:3, 3]
    R_imu_cam     = R.from_matrix(R_imu_cam_mat)
    print(f"VOXL extrinsic: t_imu_cam={t_imu_cam}  |t|={np.linalg.norm(t_imu_cam):.3f} m")

    # IMU clock → cam clock: timeshift = +15 ms → t_offset = -15 ms.
    timeshift = config.load_timeshift_imu_cam()
    t_offset  = -timeshift
    print(f"timeshift_imu_cam = {timeshift*1000:+.1f} ms  →  IMU t_offset = {t_offset*1000:+.1f} ms")

    # ---- DATA ----
    loader = ThermalDataLoader(
        config.CAM0_DIR,
        bit_depth=8,                # mono8 decoded thermal → normalize no-op
        undistort=True, K=K, D=D,
        use_clahe=USE_CLAHE,
        **PREPROC,
    )
    loader.load_timestamps_file(config.CAM0_TS, unit='s')
    imu = IMULoader(config.IMU_CSV, t_offset=t_offset)

    gt  = config.load_ground_truth()
    gps = config.load_gps()
    print(f"images: {len(loader)}    IMU: {len(imu)}    cam dur: "
          f"{loader.timestamps[-1]-loader.timestamps[0]:.1f} s")
    print(f"GT (EKF2): {len(gt['t'])} samples, {gt['t'][-1]-gt['t'][0]:.1f} s")
    print(f"GPS (raw): {len(gps['t'])} fixes @ 3D, eph_mean={gps['eph'].mean():.2f} m")

    # ---- MSCKF ----
    msckf = MSCKF(
        K=K, D=None,
        **MSCKF_PARAMS,             # shared back-end config (common_params)
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

    # ---- INIT ----
    # No static segment. Strategy (same as FIReStereo's gravity-aligned init,
    # but with a quasi-static window instead of a truly static one):
    #   - pos        : from PX4 odom's first pose
    #   - roll/pitch : from the quasi-static window's accel mean (gravity direction)
    #   - yaw        : from PX4 odom's first pose (accel cannot observe yaw)
    #   - vel        : 0
    #   - bg         : quasi-static window's gyro mean
    #   - ba         : a_mean - R_init^T @ g  (R_init is now consistent with accel)
    #
    # We previously took R_init directly from PX4 odom's q[0]; EKF2's pitch
    # estimate deviates by 5 deg from the accel-derived pitch, and that
    # deviation leaks a wrong gravity component into ba_init, causing drift.
    t_init = gt['t'][0]
    msckf.nominal_pos = gt['p'][0].copy()
    msckf.nominal_vel = np.zeros(3)
    msckf.gravity     = np.array([0.0, 0.0, config.GRAVITY_MAGNITUDE])

    # Quasi-static window istatistikleri
    t0_qs = t_init + config.INIT_QUASI_STATIC_T0
    t1_qs = t_init + config.INIT_QUASI_STATIC_T1
    mask_qs = (imu.timestamps >= t0_qs) & (imu.timestamps <= t1_qs)
    if mask_qs.sum() < 100:
        sys.exit(f"Quasi-static window'da yetersiz IMU sample ({mask_qs.sum()}); "
                 "config.INIT_QUASI_STATIC_T0/T1 kontrol et.")
    g_qs       = imu.gyro[mask_qs]
    a_qs       = imu.accel[mask_qs]
    gyro_norm_qs = np.linalg.norm(g_qs, axis=1).mean()
    a_mean_qs  = a_qs.mean(axis=0)
    a_norm_qs  = np.linalg.norm(a_mean_qs)

    # Yaw: PX4 odom'dan
    qx, qy, qz, qw = gt['q'][0]
    yaw_gt = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=True)[2]
    yaw_gt = yaw_gt + 180.0      # PX4 odom body X points opposite to IMU forward
    if yaw_gt > 180.0:
        yaw_gt -= 360.0
    # Roll/pitch: accel mean'den (gravity-aligned).
    # Convention: ROS FLU body (x forward, y left, z up). Stationary accel
    # reads a_body = R_init^T @ [0, 0, +g]. Empirically verified by checking
    # that R_init^T @ g == a_mean, |ba| ~ 0:
    #   pitch = atan2(-a_x, sqrt(a_y² + a_z²))
    #   roll  = atan2( a_y, a_z)
    # (Earlier signs were inverted; led to |ba|~1.5 m/s² fake gravity offset.)
    roll_acc  = float(np.degrees(np.arctan2( a_mean_qs[1], a_mean_qs[2])))
    pitch_acc = float(np.degrees(np.arctan2(-a_mean_qs[0],
                                              np.sqrt(a_mean_qs[1]**2 + a_mean_qs[2]**2))))
    R_init = R.from_euler('xyz', [roll_acc, pitch_acc, yaw_gt], degrees=True)
    msckf.nominal_rot     = R_init
    msckf.nominal_rot_fej = R_init

    # bg: NOT estimated from quasi-static window — gyro |w|=0.12 rad/s during
    # window dominates over typical MEMS bias (~0.005 rad/s), so the "mean
    # gyro" mostly captures actual motion rather than bias. Subtracting it
    # would cancel real rotations and keep the filter pointing straight even
    # when GT turns. Initialize to zero and let VIO update converge bias online.
    msckf.bg = config.IMU_INIT_BIAS_GYRO.copy()

    # Also include velocity from PX4 odom numerical derivative (drone is not
    # stationary at t_init; vel=0 init causes immediate XY divergence).
    dt0 = gt['t'][1] - gt['t'][0]
    if dt0 > 0:
        msckf.nominal_vel = (gt['p'][1] - gt['p'][0]) / dt0
        print(f"  vel init     : {msckf.nominal_vel}  |v|={np.linalg.norm(msckf.nominal_vel):.2f} m/s")

    g_body_expected = R_init.as_matrix().T @ msckf.gravity
    msckf.ba = config.IMU_INIT_BIAS_ACCEL.copy()

    msckf.lock_imu_attitude = LOCK_ATTITUDE
    msckf.chi2_enabled      = CHI2_GATE

    init_rpy = msckf.nominal_rot.as_euler('xyz', degrees=True)
    print(f"\nPX4-odom-init @ t={t_init:.3f}s")
    print(f"  pos          : {msckf.nominal_pos}")
    print(f"  attitude     : roll={init_rpy[0]:+.2f}°  pitch={init_rpy[1]:+.2f}°  "
          f"yaw={init_rpy[2]:+.2f}°")
    print(f"  quasi-static : [{config.INIT_QUASI_STATIC_T0}, {config.INIT_QUASI_STATIC_T1}] s  "
          f"({mask_qs.sum()} sample, |w|={gyro_norm_qs:.4f} rad/s, |a|={a_norm_qs:.3f})")
    print(f"  bg(init)     : {msckf.bg}")
    print(f"  ba(init)     : {msckf.ba}    |ba|={np.linalg.norm(msckf.ba):.4f}")

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

        gt_xy_line,  = ax_xy.plot([], [], 'r--', lw=1.3, alpha=0.85, label='EKF2 (PX4)')
        tio_xy_line, = ax_xy.plot([], [], 'b-',  lw=1.3, label='TIO')
        gt_xy_pt     = ax_xy.scatter([], [], c='r', s=70, zorder=5, edgecolor='k')
        tio_xy_pt    = ax_xy.scatter([], [], c='b', s=70, zorder=5, edgecolor='k')
        ax_xy.scatter([0.0], [0.0], c='g', s=80, marker='o', zorder=5, label='start')
        ax_xy.set_xlabel('x [m]'); ax_xy.set_ylabel('y [m]')
        ax_xy.set_aspect('equal'); ax_xy.grid(alpha=0.3); ax_xy.legend(loc='best')
        ax_xy.set_title('Top-down (XY) — bounds auto-scale')

        gt_z_line,  = ax_z.plot([], [], 'r--', lw=1.3, alpha=0.85, label='EKF2 (PX4)')
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

        fig.suptitle(f"VOXL run_v3 — MSCKF TIO ({METHOD.upper()})  |  t =   0.0 s",
                     fontsize=12)

    # ---- EVENT LOOP ----
    seq = VIOSequencer(loader, imu)
    traj       = []
    prev_imu_t = t_init
    frame_id   = 0
    n_predict  = 0
    n_frame    = 0
    prev_img_cache = None
    track_log      = []   # (t, n_active, n_used) per cam frame

    for kind, t, payload in seq.events():
        if t <= t_init:
            continue

        if kind == 'imu':
            dt = t - prev_imu_t
            if dt > 0:
                # VOXL /imu_apps quirk: accel reported in FLU body (z=+9.7),
                # gyro reported in FRD body. Convert gyro to FLU by flipping
                # Y and Z (rotation 180° about X). Verified empirically by
                # comparing gyro integration vs GT yaw/pitch over first 5s.
                gyro_fl = np.array([
                    GYRO_SIGN[0] * payload[1],
                    GYRO_SIGN[1] * payload[2],
                    GYRO_SIGN[2] * payload[3],
                ])
                accel   = payload[4:7]
                msckf.predict(dt, gyro_fl, accel)
            prev_imu_t = t
            n_predict += 1
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

        # ── LIVE PLOT REDRAW ───────────────────────────────────────────
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
            pad_xy = max(1.0, 0.1 * max(xs.ptp(), ys.ptp()))
            pad_z  = max(0.3, 0.1 * zs.ptp())
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
                f"VOXL run_v3 — MSCKF TIO ({METHOD.upper()})  |  "
                f"t = {t - t_init:5.1f} s   active tracks: {n_active}",
                fontsize=12,
            )
            plt.pause(0.001)

        prev_img_cache = img

        # GT comparison log
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
    print(f"\nmode={MODE}   processed: {n_predict} IMU predicts, {n_frame} cam frames")

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

        print(f"\nvs EKF2 (PX4 odom — pseudo-GT, shares IMU):")
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

        out_base = (f"voxlbag_v3_{METHOD.upper()}_CLAHE{'ON' if USE_CLAHE else 'OFF'}"
                    + ('' if MODE == 'tio' else f"_{MODE.upper()}"))
        out_txt = f"results/{out_base}.txt"
        out_png_traj = f"results/{out_base}_traj.png"
        out_png_diag = f"results/{out_base}_diag.png"
        out_csv = "results/all_runs.csv"
        os.makedirs('results', exist_ok=True)
        np.savetxt(out_txt, traj, header='t x y z qx qy qz qw')

        title_info = {
            'run_id':         out_base,
            'dataset':        'voxlbag_v3',
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