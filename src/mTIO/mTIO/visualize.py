import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(positions, gt_positions=None, title="Monocular Thermal VO Trajectory",
                    save_path=None):
    """
    Bird's eye view (top-down) trajectory plot.

    Camera coordinate system:
      X: right / left
      Y: down  / up
      Z: forward / backward
    Top-down view uses X and Z axes.

    Args:
        positions:    (N, 3) numpy array — estimated VO positions [x, y, z]
        gt_positions: (M, 3) numpy array — ground truth positions (optional)
        title:        plot title
        save_path:    if given, saves figure to this path instead of showing
    """
    if len(positions) == 0:
        print("No data to plot.")
        return

    x = positions[:, 0]
    z = positions[:, 2]

    plt.figure(figsize=(10, 8))

    plt.plot(x, z, marker='.', markersize=2, linestyle='-',
             color='steelblue', label='VO Trajectory', alpha=0.8)
    plt.plot(x[0],  z[0],  'go', markersize=8, label='Start')
    plt.plot(x[-1], z[-1], 'ro', markersize=8, label='End')

    if gt_positions is not None and len(gt_positions) > 0:
        gx = gt_positions[:, 0]
        gy = gt_positions[:, 1]   # GT is typically X-Y (world frame)
        plt.plot(gx, gy, linestyle='--', color='orange',
                 linewidth=1.5, label='Ground Truth', alpha=0.8)
        plt.plot(gx[0],  gy[0],  'g^', markersize=8)
        plt.plot(gx[-1], gy[-1], 'r^', markersize=8)

    plt.title(title)
    plt.xlabel('X (right / left)  [unitless]')
    plt.ylabel('Z (forward / back) [unitless]')
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()
