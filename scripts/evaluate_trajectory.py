#!/usr/bin/env python3
"""
VO vs Ground Truth trajectory comparison.
- Timestamp-based nearest-neighbor matching
- Sim(3) alignment (scale + rotation + translation)
- ATE (Absolute Trajectory Error) computation
- Single figure: 2D top-down plot + error stats
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def load_tum_trajectory(filepath):
    """Load TUM-format trajectory: timestamp tx ty tz qx qy qz qw"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                data.append([float(p) for p in parts[:8]])
    data = np.array(data)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quaternions = data[:, 4:8]  # qx qy qz qw
    return timestamps, positions, quaternions


def associate_trajectories(ts_gt, ts_vo, max_diff=0.05):
    """Find nearest-neighbor timestamp matches between GT and VO."""
    matches = []
    for i, t_vo in enumerate(ts_vo):
        idx = np.argmin(np.abs(ts_gt - t_vo))
        if abs(ts_gt[idx] - t_vo) < max_diff:
            matches.append((idx, i))
    return np.array(matches)


def sim3_align(pos_gt, pos_vo):
    """
    Sim(3) alignment: find scale s, rotation R, translation t
    such that pos_gt ≈ s * R @ pos_vo + t
    Uses Umeyama's method.
    """
    mu_gt = pos_gt.mean(axis=0)
    mu_vo = pos_vo.mean(axis=0)

    gt_centered = pos_gt - mu_gt
    vo_centered = pos_vo - mu_vo

    # Cross-covariance
    H = vo_centered.T @ gt_centered / len(pos_gt)

    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    # Scale
    var_vo = np.sum(vo_centered ** 2) / len(pos_vo)
    scale = np.trace(np.diag([1, 1, d]) @ np.diag(S)) / var_vo

    # Translation
    t = mu_gt - scale * R @ mu_vo

    return scale, R, t


def apply_sim3(positions, scale, R, t):
    """Apply Sim(3) transform to positions."""
    return (scale * (R @ positions.T)).T + t


def compute_ate(pos_gt, pos_vo_aligned):
    """Compute ATE (RMSE of position differences)."""
    errors = np.linalg.norm(pos_gt - pos_vo_aligned, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    return errors, rmse


def main():
    gt_path = "/home/akin/firestereo/firestereo/reconstruction/hawkins_run4_car/traj.txt"
    vo_path = "/tmp/vo_trajectory.txt"
    output_path = "/home/akin/tvio_ws/results/trajectory_comparison.png"

    if len(sys.argv) >= 3:
        gt_path = sys.argv[1]
        vo_path = sys.argv[2]
    if len(sys.argv) >= 4:
        output_path = sys.argv[3]

    print(f"GT:  {gt_path}")
    print(f"VO:  {vo_path}")

    # Load
    ts_gt, pos_gt, quat_gt = load_tum_trajectory(gt_path)
    ts_vo, pos_vo, quat_vo = load_tum_trajectory(vo_path)
    print(f"GT poses: {len(ts_gt)}, VO poses: {len(ts_vo)}")

    # Associate by timestamp
    matches = associate_trajectories(ts_gt, ts_vo, max_diff=0.05)
    print(f"Matched poses: {len(matches)}")

    if len(matches) < 10:
        print("ERROR: Too few matches. Check timestamp alignment.")
        print(f"  GT time range:  [{ts_gt[0]:.3f}, {ts_gt[-1]:.3f}]")
        print(f"  VO time range:  [{ts_vo[0]:.3f}, {ts_vo[-1]:.3f}]")
        sys.exit(1)

    pos_gt_matched = pos_gt[matches[:, 0]]
    pos_vo_matched = pos_vo[matches[:, 1]]

    # Sim(3) alignment
    scale, R, t = sim3_align(pos_gt_matched, pos_vo_matched)
    print(f"Alignment scale: {scale:.4f}")

    pos_vo_aligned = apply_sim3(pos_vo_matched, scale, R, t)
    pos_vo_full_aligned = apply_sim3(pos_vo, scale, R, t)

    # ATE
    errors, ate_rmse = compute_ate(pos_gt_matched, pos_vo_aligned)
    ate_mean = np.mean(errors)
    ate_median = np.median(errors)
    ate_max = np.max(errors)
    ate_std = np.std(errors)

    print(f"\n--- ATE (Absolute Trajectory Error) ---")
    print(f"  RMSE:   {ate_rmse:.4f} m")
    print(f"  Mean:   {ate_mean:.4f} m")
    print(f"  Median: {ate_median:.4f} m")
    print(f"  Std:    {ate_std:.4f} m")
    print(f"  Max:    {ate_max:.4f} m")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Top-down view (X-Y)
    ax1 = axes[0]
    ax1.plot(pos_gt[:, 0], pos_gt[:, 1], 'b-', linewidth=1.5, label='Ground Truth', alpha=0.8)
    ax1.plot(pos_vo_full_aligned[:, 0], pos_vo_full_aligned[:, 1], 'r-', linewidth=1.5, label='VO (aligned)', alpha=0.8)
    ax1.plot(pos_gt[0, 0], pos_gt[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(pos_gt[-1, 0], pos_gt[-1, 1], 'ks', markersize=10, label='End (GT)')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('Top-Down View (X-Y)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # ATE over time
    ax2 = axes[1]
    matched_ts = ts_vo[matches[:, 1]]
    time_relative = matched_ts - matched_ts[0]
    ax2.plot(time_relative, errors, 'r-', linewidth=1, alpha=0.7)
    ax2.axhline(y=ate_rmse, color='k', linestyle='--', linewidth=1, label=f'RMSE = {ate_rmse:.3f} m')
    ax2.fill_between(time_relative, 0, errors, alpha=0.2, color='red')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Position Error (m)', fontsize=12)
    ax2.set_title('ATE Over Time', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Stats text box
    stats_text = (
        f"ATE RMSE: {ate_rmse:.4f} m\n"
        f"ATE Mean: {ate_mean:.4f} m\n"
        f"ATE Max:  {ate_max:.4f} m\n"
        f"Scale:    {scale:.4f}\n"
        f"Matched:  {len(matches)} poses"
    )
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=11,
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('VO Trajectory vs Ground Truth — hawkins_4', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()