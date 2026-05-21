"""
End-to-end smoke test for the thermal MSCKF VIO pipeline.

Pipeline wiring:
    IMU event  → MSCKF.predict
    CAM event  → MSCKF.augment_state
                 ORBTracker.process_frame
                 MSCKF.update(dead_tracks)
                 MSCKF.prune_cam_states

This is a sanity check, not a tuned run. The update path runs the full
Mourikis pipeline — DLT triangulation, inverse-depth Gauss-Newton
refinement, parallax + chi-square gating, and a sequential per-track
Joseph update — but the parameters here are not tuned, so drift on
low-parallax segments is still expected. The point is to verify the full
chain runs end-to-end and the trajectory is plausible.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/thermal_vo'))

from thermal_vo.dataloader      import ThermalDataLoader, IMULoader, VIOSequencer
from thermal_vo.orb             import ORBTracker
from thermal_vo.msckf           import MSCKF
from thermal_vo                 import config_sthereo as config


MAX_WINDOW = 15


def run():
    # ---- CALIBRATION ----
    cal = config.load_camera_intrinsics()
    K, D = cal['K'], cal['D']

    # SThereo's imu_2_thermal_left stores T_cam_imu (maps IMU → cam).
    # MSCKF.augment_state wants R_imu_cam (maps cam vectors → IMU frame)
    # and t_imu_cam (cam origin in IMU frame), so invert:
    T_cam_imu  = config.load_extrinsic()
    R_cam_imu  = T_cam_imu[:3, :3]
    t_cam_imu  = T_cam_imu[:3, 3]
    R_imu_cam_mat = R_cam_imu.T
    t_imu_cam     = -R_imu_cam_mat @ t_cam_imu
    R_imu_cam     = R.from_matrix(R_imu_cam_mat)

    # ---- DATA ----
    # Image-level undistortion is done in the loader, so MSCKF receives
    # already-pinhole pixels — pass D=None to pixels_to_normalized.
    loader = ThermalDataLoader(
        config.THERMAL_LEFT_DIR,
        bit_depth=16,
        undistort=True, K=K, D=D,
    )
    imu = IMULoader.sthereo(config.IMU_CSV)

    print(f"images: {len(loader)}    IMU: {len(imu)}")

    # ---- STATIC INIT ----
    n_static = int(config.STATIC_INIT_SECONDS / np.median(np.diff(imu.timestamps)))
    msckf = MSCKF(K=K, D=None, pixel_noise_std=3.0, min_parallax_deg=0.15, min_depth=0.1, max_depth=500.0,
                 gn_max_iter=10, chi2_alpha=0.99,
                 init_att_std=0.02, init_bg_std=0.05, init_vel_std=0.5, init_ba_std=0.05, init_pos_std=0.05
                 )
    msckf.initialize_from_static(imu.gyro[:n_static], imu.accel[:n_static])
    t_init_end = imu.timestamps[n_static - 1]

    g_mag      = np.linalg.norm(msckf.gravity)
    init_euler = msckf.nominal_rot.as_euler('xyz', degrees=True)
    print(f"static init: {n_static} samples,  |g|={g_mag:.4f},  "
          f"RPY=({init_euler[0]:+.2f}, {init_euler[1]:+.2f}, {init_euler[2]:+.2f})°")

    # ---- TRACKER ----
    tracker = ORBTracker(
        n_features=400, grid_rows=4, grid_cols=4,
        ratio_thresh=0.75, ransac_thresh=2.0,
        min_track_length=3,
    )

    # ---- EVENT LOOP ----
    seq = VIOSequencer(loader, imu)
    traj = []
    prev_imu_t = t_init_end
    frame_id   = 0
    n_predict  = 0
    n_frame    = 0

    for kind, t, payload in seq.events():
        if t <= t_init_end:
            continue

        if kind == 'imu':
            dt = t - prev_imu_t
            if dt > 0:
                gyro = payload[1:4]
                acc  = payload[4:7]
                msckf.predict(dt, gyro, acc)
            prev_imu_t = t
            n_predict += 1
            continue

        # kind == 'cam'
        img = payload
        msckf.augment_state(frame_id, R_imu_cam, t_imu_cam)
        tracker.process_frame(img, frame_id)
        msckf.update(tracker.dead_tracks)
        msckf.prune_cam_states(MAX_WINDOW)

        traj.append((t, *msckf.nominal_pos))
        frame_id += 1
        n_frame  += 1

        if n_frame % 50 == 0:
            print(f"  frame {n_frame:4d}  pos=({msckf.nominal_pos[0]:+.2f}, "
                  f"{msckf.nominal_pos[1]:+.2f}, {msckf.nominal_pos[2]:+.2f})  "
                  f"win={len(msckf.cam_states)}  "
                  f"retired={len(tracker.dead_tracks)}")

    traj = np.array(traj)
    print(f"\nprocessed: {n_predict} IMU predicts, {n_frame} frames")
    print(f"final pos: ({traj[-1, 1]:+.3f}, {traj[-1, 2]:+.3f}, {traj[-1, 3]:+.3f})")

    # ---- PERSIST ----
    os.makedirs('results', exist_ok=True)
    out_txt = 'results/msckf_vio_trajectory.txt'
    np.savetxt(out_txt, traj, header='t x y z')
    print(f"saved → {out_txt}")

    # ---- COMPARE TO GT ----
    gt   = np.loadtxt(config.GT_LOCAL_POSE, delimiter=',')
    gt_t = gt[:, 0]; gt_xyz = gt[:, 1:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(traj[:, 1], traj[:, 2], 'C0', label='MSCKF VIO')
    axes[0, 0].plot(gt_xyz[:, 0], gt_xyz[:, 1], 'r--', label='GT')
    axes[0, 0].scatter([0], [0], c='k', s=30, zorder=5, label='start')
    axes[0, 0].set_xlabel('x [m]'); axes[0, 0].set_ylabel('y [m]')
    axes[0, 0].set_title('Top-down (XY)')
    axes[0, 0].legend(); axes[0, 0].axis('equal'); axes[0, 0].grid(alpha=0.3)

    t0 = traj[0, 0]
    for ax, axis_idx, label in zip(axes.flat[1:], [1, 2, 3], ['x', 'y', 'z']):
        ax.plot(traj[:, 0] - t0, traj[:, axis_idx], 'C0', label='VIO')
        ax.plot(gt_t  - t0,      gt_xyz[:, axis_idx - 1], 'r--', label='GT')
        ax.set_xlabel('t [s]'); ax.set_ylabel(f'{label} [m]')
        ax.set_title(f'{label} vs time'); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = 'results/msckf_vio_test.png'
    plt.savefig(out_png, dpi=120)
    print(f"plot saved → {out_png}")


if __name__ == '__main__':
    run()
