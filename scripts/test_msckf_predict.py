"""
Predict-only sanity check for the MSCKF backbone.
  1. Load Xsens IMU.
  2. Use the first STATIC_INIT_SECONDS of samples (assumed stationary) to
     estimate gyro bias and the initial gravity-aligned attitude.
  3. Run predict() on every remaining IMU sample.
  4. Save the integrated trajectory and overlay it against the local-pose
     ground truth.

Without visual updates the trajectory will drift quickly — that is expected.
The point of this script is to verify that the predict step is well-formed,
that the static initialization is sane, and to give us a baseline for the
upcoming visual update step.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/thermal_vo'))

from thermal_vo.dataloader import IMULoader
from thermal_vo.msckf      import MSCKF
from thermal_vo            import config_sthereo as config


def run():
    imu = IMULoader.sthereo(config.IMU_CSV)
    print(f"loaded {len(imu)} IMU samples, {imu.timestamps[-1] - imu.timestamps[0]:.1f}s span")

    # Static initialization
    n_static = int(config.STATIC_INIT_SECONDS / np.median(np.diff(imu.timestamps)))
    msckf = MSCKF()
    msckf.initialize_from_static(imu.gyro[:n_static], imu.accel[:n_static])

    g_mag = np.linalg.norm(msckf.gravity)
    init_euler = msckf.nominal_rot.as_euler('xyz', degrees=True)
    print(f"static init: {n_static} samples")
    print(f"  bg estimate : {msckf.bg}")
    print(f"  |g|         : {g_mag:.4f} m/s²   (datasheet ~9.81)")
    print(f"  init RPY    : roll={init_euler[0]:+.2f}°  pitch={init_euler[1]:+.2f}°  yaw={init_euler[2]:+.2f}°")

    # Predict on the rest
    traj = np.empty((len(imu) - n_static, 4))
    prev_t = imu.timestamps[n_static - 1]
    for k, i in enumerate(range(n_static, len(imu))):
        t = imu.timestamps[i]
        dt = t - prev_t
        prev_t = t
        msckf.predict(dt, imu.gyro[i], imu.accel[i])
        traj[k] = (t, *msckf.nominal_pos)

    # Persist
    os.makedirs('results', exist_ok=True)
    out_txt = 'results/predict_only_trajectory.txt'
    np.savetxt(out_txt, traj, header='t x y z')
    print(f"\nfinal position: {traj[-1, 1:]}")
    print(f"trajectory saved → {out_txt}")

    # Ground truth (local frame)
    gt = np.loadtxt(config.GT_LOCAL_POSE, delimiter=',')
    gt_t  = gt[:, 0]
    gt_xyz = gt[:, 1:4]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(traj[:, 1], traj[:, 2], label='MSCKF predict-only', color='C0')
    axes[0, 0].plot(gt_xyz[:, 0], gt_xyz[:, 1], 'r--', label='GT (local_pose)')
    axes[0, 0].scatter([0], [0], c='k', marker='o', s=30, zorder=5, label='start')
    axes[0, 0].set_xlabel('x [m]'); axes[0, 0].set_ylabel('y [m]')
    axes[0, 0].set_title('Top-down trajectory')
    axes[0, 0].legend(); axes[0, 0].axis('equal'); axes[0, 0].grid(alpha=0.3)

    t0 = traj[0, 0]
    for ax, axis_idx, label in zip(axes.flat[1:], [1, 2, 3], ['x', 'y', 'z']):
        ax.plot(traj[:, 0] - t0, traj[:, axis_idx], label='predict', color='C0')
        ax.plot(gt_t - t0, gt_xyz[:, axis_idx - 1], 'r--', label='GT')
        ax.set_xlabel('time [s]'); ax.set_ylabel(f'{label} [m]')
        ax.set_title(f'{label} vs time'); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = 'results/predict_only.png'
    plt.savefig(out_png, dpi=120)
    print(f"plot saved      → {out_png}")


if __name__ == '__main__':
    run()
