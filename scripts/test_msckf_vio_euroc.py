"""
EuRoC MAV (MH_03_medium) — MSCKF VIO validation.

Tests whether the STheReo bug hunt was dataset-independent. The same MSCKF
(PHASE 2 fix + FEJ landmark anchoring + IMU-state FEJ rot-only) is run on
EuRoC as well. If the pipeline is correct, it should give a clean result on
EuRoC; if it still diverges, there is still work left in the pipeline.

Key differences on EuRoC:
  * Body frame == IMU frame -> cam0/sensor.yaml's T_BS IS DIRECTLY T_imu_cam.
  * Images are 8-bit grayscale, no CLAHE/blur needed.
  * Init: from GT's first sample (instead of static-init, since the drone is
    already moving at the start).
  * IMU noise: ADIS16448 (cleaner than the Xsens units).

Run:
    python3 scripts/test_msckf_vio_euroc.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/thermal_vo'))

DL_VER = os.environ.get('TVIO_DL_VER', '1')
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
from thermal_vo            import config_euroc as config
from thermal_vo.common_params import *   # shared switches, window, ZUPT, PREPROC, MSCKF_PARAMS, make_tracker


# ── FRONT-END ─────────────────────────────────────────────────────────────
METHOD     = os.environ.get('TVIO_METHOD', 'orb')        # 'klt' or 'orb'

# ── ODOMETRY MODE ─────────────────────────────────────────────────────────
MODE       = 'vio'        # 'vio' (full fusion) or 'imu' (pure dead-reckoning)
if MODE not in ('vio', 'imu'):
    raise ValueError(f"Unknown MODE={MODE!r}")

# ── THERMAL PREPROCESSING ─────────────────────────────────────────────────
# EuRoC 8-bit RGB baseline: percentile normalisation is a no-op (bit_depth==8),
# CLAHE defaults to off (the image is already clean). Uses the same shared
# PREPROC settings (common_params) as the other datasets.
USE_CLAHE      = os.environ.get('TVIO_USE_CLAHE', '0') == '1'
GAUSSIAN_SIGMA = PREPROC['gaussian_sigma']

# Shared diagnostic switches (LOCK_ATTITUDE, CHI2_GATE), sliding-window sizes
# (MAX_WINDOW, MAX_TRACKS_PER_UPDATE) and the full ZUPT block come from
# `common_params` via the star import above.

# ── LIVE PLOT ─────────────────────────────────────────────────────────────
# True: live-updates two plots (top-down XY + Z(t)) while the run is going.
# False: run silently, save a PNG at the end (for headless / Agg backend).
LIVE_PLOT = os.environ.get('TVIO_LIVE_PLOT', '0') == '1'
VIZ_EVERY = 2       # redraw every Nth camera frame


def wrap_yaw_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def is_stationary(imu_buffer, gyro_var_thresh, accel_var_thresh):
    """
    Variance-based stationarity detection.
    Looks at how much the data fluctuates, not its absolute magnitude, so it
    is unaffected by static biases.
    """
    if len(imu_buffer) < ZUPT_IMU_WINDOW:
        return False

    arr = np.asarray(imu_buffer)

    # Compute each axis's (X, Y, Z) own variance
    gyro_vars = np.var(arr[:, 1:4], axis=0)
    accel_vars = np.var(arr[:, 4:7], axis=0)

    # If any axis's fluctuation exceeds the threshold, there is motion
    # (e.g. we use 'max' to also catch pure-yaw rotation)
    is_gyro_static = np.max(gyro_vars) < gyro_var_thresh
    is_accel_static = np.max(accel_vars) < accel_var_thresh
    
    return is_gyro_static and is_accel_static


def run():
    # ---- CALIBRATION ----
    cal = config.load_camera_intrinsics()
    K, D = cal['K'], cal['D']

    # EuRoC: T_BS = T_imu_cam DIRECTLY (body frame == IMU frame).
    T_imu_cam     = config.load_extrinsic()
    R_imu_cam_mat = T_imu_cam[:3, :3]
    t_imu_cam     = T_imu_cam[:3, 3]
    R_imu_cam     = R.from_matrix(R_imu_cam_mat)
    print(f"EuRoC extrinsic: t_imu_cam={t_imu_cam}  "
          f"|t|={np.linalg.norm(t_imu_cam):.3f} m")

    # ---- DATA ----
    # EuRoC: 8-bit grayscale, no CLAHE/blur needed (RGB is already clean).
    loader = ThermalDataLoader(
        config.CAM0_DIR,
        bit_depth=8,                # 8-bit -> normalize/percentile is a no-op
        undistort=True, K=K, D=D,
        use_clahe=USE_CLAHE,
        **PREPROC,
    )
    imu = IMULoader(config.IMU_CSV, delimiter=',', skip_header=0)
    print(f"images: {len(loader)}    IMU: {len(imu)}")

    gt = config.load_ground_truth()
    print(f"GT samples: {len(gt['t'])}, t = [{gt['t'][0]:.3f}, {gt['t'][-1]:.3f}] s")

    # ---- MSCKF ----
    # EuRoC RGB tracking is far more robust than thermal -> tighter
    # parameters. Since init comes from GT, init_std values are small (so
    # the filter starts confidently).
    msckf = MSCKF(
        K=K, D=None,                # undistortion already done in the loader
        **MSCKF_PARAMS,             # shared back-end config (common_params)
    )

    # Rebuild the Q matrix for EuRoC's ADIS16448 (the constructor left the
    # Xsens defaults in place).
    vg = config.IMU_GYRO_NOISE_DENSITY  ** 2
    va = config.IMU_ACCEL_NOISE_DENSITY ** 2
    vbg = config.IMU_GYRO_BIAS_RW       ** 2
    vba = config.IMU_ACCEL_BIAS_RW      ** 2
    msckf.Q_matrix = np.diag([
        vg, vg, vg,
        vbg, vbg, vbg,
        va, va, va,
        vba, vba, vba,
        0.0, 0.0, 0.0,
    ])

    # ---- GT-BASED INIT ----
    # The drone is already moving at the start of MH_03; static-init is not
    # reliable. Build the nominal state from GT's first sample, set
    # t_init = GT[0].t, and skip earlier events.
    msckf.nominal_pos = gt['p'][0].copy()
    msckf.nominal_vel = gt['v'][0].copy()
    qw, qx, qy, qz    = gt['q'][0]
    msckf.nominal_rot = R.from_quat([qx, qy, qz, qw])  # scipy: [x,y,z,w]
    msckf.bg          = gt['bg'][0].copy()
    msckf.ba          = gt['ba'][0].copy()
    msckf.gravity     = np.array([0.0, 0.0, config.GRAVITY_MAGNITUDE])
    # Align the IMU-state FEJ anchor to the init too.
    msckf.nominal_rot_fej = msckf.nominal_rot

    msckf.lock_imu_attitude = LOCK_ATTITUDE
    msckf.chi2_enabled      = CHI2_GATE

    t_init = gt['t'][0]
    init_rpy = msckf.nominal_rot.as_euler('xyz', degrees=True)
    msckf_yaw_init_deg = init_rpy[2]
    print(f"GT-init @ t={t_init:.3f}s  pos={msckf.nominal_pos}  "
          f"rpy=({init_rpy[0]:+.2f}, {init_rpy[1]:+.2f}, {init_rpy[2]:+.2f})°")
    print(f"  bg={msckf.bg}  ba={msckf.ba}")

    # ---- TRACKER ----
    tracker = make_tracker(METHOD)
    print(f"front-end: {METHOD.upper()}  mode={MODE}  CHI2={CHI2_GATE}  "
          f"LOCK_ATT={LOCK_ATTITUDE}  max_tracks/upd={MAX_TRACKS_PER_UPDATE}")

    # ---- LIVE PLOT SETUP (opsiyonel) ─────────────────────────────────────
    gt_x, gt_y, gt_z = gt['p'][:, 0], gt['p'][:, 1], gt['p'][:, 2]
    if LIVE_PLOT:
        plt.ion()
        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
        ax_xy  = fig.add_subplot(gs[0, 0])
        ax_z   = fig.add_subplot(gs[0, 1])
        ax_img = fig.add_subplot(gs[1, :])
        gt_t_rel = gt['t'] - t_init

        # Both the GT and VIO lines start EMPTY — each redraw plots the
        # portion accumulated so far. Axis limits are fixed up front to GT's
        # full range, so the view doesn't keep shifting.
        gt_xy_line,  = ax_xy.plot([], [], 'r--', lw=1.3, alpha=0.85, label='GT')
        vio_xy_line, = ax_xy.plot([], [], 'b-',  lw=1.3, label='VIO')
        gt_xy_pt     = ax_xy.scatter([], [], c='r', s=70, zorder=5, edgecolor='k')
        vio_xy_pt    = ax_xy.scatter([], [], c='b', s=70, zorder=5, edgecolor='k')
        ax_xy.scatter([0.0], [0.0], c='g', s=80, marker='o', zorder=5, label='start')
        pad_xy = 0.5
        ax_xy.set_xlim(gt_x.min() - pad_xy, gt_x.max() + pad_xy)
        ax_xy.set_ylim(gt_y.min() - pad_xy, gt_y.max() + pad_xy)
        ax_xy.set_xlabel('x [m]'); ax_xy.set_ylabel('y [m]')
        ax_xy.set_aspect('equal'); ax_xy.grid(alpha=0.3); ax_xy.legend(loc='best')
        ax_xy.set_title('Top-down (XY)')

        gt_z_line,  = ax_z.plot([], [], 'r--', lw=1.3, alpha=0.85, label='GT')
        vio_z_line, = ax_z.plot([], [], 'b-',  lw=1.3, label='VIO')
        gt_z_pt     = ax_z.scatter([], [], c='r', s=70, zorder=5, edgecolor='k')
        vio_z_pt    = ax_z.scatter([], [], c='b', s=70, zorder=5, edgecolor='k')
        pad_z = 0.3
        ax_z.set_xlim(gt_t_rel.min(), gt_t_rel.max())
        ax_z.set_ylim(gt_z.min() - pad_z, gt_z.max() + pad_z)
        ax_z.set_xlabel('t [s]'); ax_z.set_ylabel('z [m]')
        ax_z.grid(alpha=0.3); ax_z.legend(loc='best')
        ax_z.set_title('Height (Z) over time')

        fig.suptitle(f"EuRoC MH_03 — MSCKF VIO ({METHOD.upper()})  |  t =   0.0 s",
                     fontsize=12)

    # ---- EVENT LOOP ----
    seq = VIOSequencer(loader, imu)
    traj       = []
    prev_imu_t = t_init
    frame_id   = 0
    n_predict  = 0
    n_frame    = 0
    n_zupt     = 0
    imu_buffer = []     # last N IMU samples, for ZUPT
    zupt_ctr   = 0      # ZUPT throttle counter
    is_airborne = False # take-off latch — once True, never goes back to False
    track_log   = []    # (t, n_active, n_used) per cam frame, for 4-panel plot

    for kind, t, payload in seq.events():
        if t <= t_init:
            continue

        if kind == 'imu':
            dt = t - prev_imu_t
            if dt > 0:
                msckf.predict(dt, payload[1:4], payload[4:7])
            prev_imu_t = t
            n_predict += 1

            # ── ZUPT: feed a pseudo "velocity = 0" measurement when at rest ─────
            if ZUPT:
                imu_buffer.append(payload)
                if len(imu_buffer) > ZUPT_IMU_WINDOW:
                    imu_buffer.pop(0)
                zupt_ctr += 1
                if zupt_ctr % ZUPT_THROTTLE == 0:
                    if is_stationary(imu_buffer, ZUPT_GYRO_THRESH,
                                     ZUPT_ACCEL_DEV_THRESH):
                        msckf.zero_velocity_update(sigma_zupt=ZUPT_SIGMA)
                        n_zupt += 1
            continue

        # kind == 'cam'
        img = payload

        if MODE == 'vio':
            msckf.now = t                    # timestamp for the NIS log
            msckf.augment_state(frame_id, R_imu_cam, t_imu_cam)
            tracker.process_frame(img, frame_id)

            n_drop = len(msckf.cam_states) - MAX_WINDOW
            if n_drop > 0:
                for cs in msckf.cam_states[:n_drop]:
                    tracker.marginalize_at_prune(cs.frame_id)

            n_retired = len(tracker.dead_tracks)
            upd_tracks = list(tracker.dead_tracks)
            if len(upd_tracks) > MAX_TRACKS_PER_UPDATE:
                upd_tracks.sort(key=lambda t_: len(t_.frame_ids), reverse=True)
                upd_tracks = upd_tracks[:MAX_TRACKS_PER_UPDATE]

            msckf.update(upd_tracks)
            msckf.prune_cam_states(MAX_WINDOW)
        else:
            n_retired = 0

        traj.append((t, *msckf.nominal_pos, *msckf.nominal_rot.as_quat()))
        n_active = len(tracker.active_tracks) if MODE == 'vio' else 0
        track_log.append((t, n_active,
                          msckf.last_n_used if MODE == 'vio' else 0))
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
            # VIO: all points accumulated so far + a marker at the last point
            vio_xy_line.set_data(vp_disp[:, 0], vp_disp[:, 1])
            vio_xy_pt.set_offsets([[vp_disp[-1, 0], vp_disp[-1, 1]]])
            vio_z_line.set_data(vt_rel, vp_disp[:, 2])
            vio_z_pt.set_offsets([[vt_rel[-1], vp_disp[-1, 2]]])
            # GT: only up to the current cam-frame time (t)
            gt_mask = gt['t'] <= t
            gt_disp_x = gt_x[gt_mask] - gt_orig[0]
            gt_disp_y = gt_y[gt_mask] - gt_orig[1]
            gt_disp_z = gt_z[gt_mask] - gt_orig[2]
            gt_xy_line.set_data(gt_disp_x, gt_disp_y)
            gt_z_line.set_data(gt_t_rel[gt_mask], gt_disp_z)
            # Red marker at the GT tip (interpolated to the current instant)
            gx_now = float(np.interp(t, gt['t'], gt_x)) - gt_orig[0]
            gy_now = float(np.interp(t, gt['t'], gt_y)) - gt_orig[1]
            gz_now = float(np.interp(t, gt['t'], gt_z)) - gt_orig[2]
            gt_xy_pt.set_offsets([[gx_now, gy_now]])
            gt_z_pt.set_offsets([[t - t_init, gz_now]])
            # Dynamic limits: let the plot auto-expand once VIO leaves the frame.
            xs = np.concatenate([gt_disp_x, vp_disp[:, 0]])
            ys = np.concatenate([gt_disp_y, vp_disp[:, 1]])
            zs = np.concatenate([gt_disp_z, vp_disp[:, 2]])
            pad_xy_dyn = max(0.5, 0.1 * max(xs.ptp(), ys.ptp()))
            pad_z_dyn  = max(0.3, 0.1 * zs.ptp())
            ax_xy.set_xlim(xs.min() - pad_xy_dyn, xs.max() + pad_xy_dyn)
            ax_xy.set_ylim(ys.min() - pad_xy_dyn, ys.max() + pad_xy_dyn)
            ax_z.set_ylim(zs.min() - pad_z_dyn, zs.max() + pad_z_dyn)
            # Current elapsed time (since t_init) in the suptitle
            fig.suptitle(
                f"EuRoC MH_03 — MSCKF VIO ({METHOD.upper()})  |  "
                f"t = {t - t_init:5.1f} s",
                fontsize=12,
            )
            plt.pause(0.001)

        # GT comparison log — expensive per-frame, so only the first 40 + every 25th.
        if n_frame <= 40 or n_frame % 25 == 0:
            i = int(np.searchsorted(gt['t'], t))
            if 0 < i < len(gt['t']):
                gp = gt['p'][i]
                gq = gt['q'][i]
                gt_R = R.from_quat([gq[1], gq[2], gq[3], gq[0]])
                eu_m = msckf.nominal_rot.as_euler('xyz', degrees=True)
                eu_g = gt_R.as_euler('xyz', degrees=True)
                err_pos   = float(np.linalg.norm(msckf.nominal_pos - gp))
                roll_err  = wrap_yaw_deg(eu_m[0] - eu_g[0])
                pitch_err = wrap_yaw_deg(eu_m[1] - eu_g[1])
                yaw_err   = wrap_yaw_deg(eu_m[2] - eu_g[2])
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

        print(f"\nvs EuRoC mocap GT (rigid SE(3) align):")
        print(f"  ATE RMSE      : {metrics['ate_rmse_m']:.3f} m")
        print(f"  ATE max       : {metrics['ate_max_m']:.3f} m")
        print(f"  drift rate    : {metrics['drift_pct']:.2f} %")
        print(f"  survival      : {metrics['survival_s']:.1f} s (threshold 0.5 m)")
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

        out_base = (f"euroc_mh03_{METHOD.upper()}_CLAHE{'ON' if USE_CLAHE else 'OFF'}"
                    + ('' if MODE == 'vio' else f"_{MODE.upper()}"))
        out_txt = f"results/{out_base}.txt"
        out_png_traj = f"results/{out_base}_traj.png"
        out_png_diag = f"results/{out_base}_diag.png"
        out_csv = "results/all_runs.csv"
        os.makedirs('results', exist_ok=True)
        np.savetxt(out_txt, traj, header='t x y z qx qy qz qw')

        title_info = {
            'run_id':         out_base,
            'dataset':        'euroc_mh03',
            'method':         METHOD,
            'mode':           MODE,
            'clahe':          False,
            'gaussian_sigma': 0.0,
            'normalize':      'plain8bit',
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

    # If live plot is on, keep the window open after the run until it's closed.
    if LIVE_PLOT:
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    run()