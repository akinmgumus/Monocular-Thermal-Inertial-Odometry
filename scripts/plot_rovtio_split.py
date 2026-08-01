"""ROVTIO alt1 — splits the saved trajectory in two at the divergence point
and plots each half separately.

Regimes (from the raw error analysis against GT):
  * seg1: t = 0 .. SPLIT_S    -> follows the shape (bounded ~5-8 m offset)
  * seg2: t = SPLIT_S .. end  -> catastrophic divergence via IMU dead-reckoning

Each segment is SE(3)-aligned independently (with_scale=False), so the rigid
offset is removed in seg1 and the motion-shape overlap is shown fairly. In
seg2 the alignment is not meaningful, but it shows the trajectory as-is
(exhibiting the divergence).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/mTIO'))

from mTIO import config_rovtio as config
from mTIO.evaluation import align_and_compute_metrics, plot_trajectory

TRAJ_TXT = 'results/msckf_vio_rovtio_alt1_orb_claheon.txt'
SPLIT_S  = 118.2          # divergence onset (the point where err>5 m becomes permanent)
OUT_BASE = 'results/msckf_vio_rovtio_alt1_orb_claheon'


def main():
    d  = np.loadtxt(TRAJ_TXT)              # t x y z
    t  = d[:, 0]
    t_rel = t - t[0]
    gt = config.load_ground_truth()

    segments = [
        ('seg1_0-%ds'      % int(SPLIT_S), t_rel <= SPLIT_S),
        ('seg2_%ds-end'    % int(SPLIT_S), t_rel >  SPLIT_S),
    ]

    for tag, mask in segments:
        seg = d[mask]
        if len(seg) < 5:
            print(f"[{tag}] not enough points ({len(seg)}), skipped")
            continue

        # Also crop GT to the segment's time window (+-1 s margin), so the
        # shape comparison is fair over the same time span.
        gmask = (gt['t'] >= seg[0, 0] - 1.0) & (gt['t'] <= seg[-1, 0] + 1.0)
        gt_t, gt_p = gt['t'][gmask], gt['p'][gmask]

        metrics = align_and_compute_metrics(seg, gt_t, gt_p,
                                            threshold_m=2.0, with_scale=False)
        title_info = {
            'dataset':        'rovtio_alt1 [%s]' % tag,
            'method':         'orb',
            'mode':           'vio',
            'clahe':          True,
            'gaussian_sigma': 1.0,
            'normalize':      'percentile',
            'chi2_alpha':     0.99,
            'max_window':     20,
        }
        out_png = f"{OUT_BASE}_{tag}_traj.png"
        plot_trajectory(seg, gt_t, gt_p, metrics,
                        title_info=title_info, out_png=out_png)

        print(f"[{tag}]  n={len(seg):5d}  dur={metrics['duration_s']:6.1f}s  "
              f"ATE_rmse={metrics['ate_rmse_m']:7.2f} m  "
              f"ATE_max={metrics['ate_max_m']:8.2f} m  "
              f"drift={metrics['drift_pct']:5.2f}%  "
              f"→ {out_png}")


if __name__ == '__main__':
    main()