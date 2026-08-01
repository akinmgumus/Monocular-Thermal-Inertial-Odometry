"""Regenerate the STheReo ORB (CLAHE on, Norm->CLAHE) tracking-window trajectory
figure WITH the whole-run GT route overlaid, so the failure-mode section can show
where on the true route the estimate breaks off (~t=95 s) and what GT was doing at
that instant. Reuses the shared posyaw alignment + plot_trajectory (t_max at the
divergence mark, gt_full_xyz = whole-run GT). Does NOT touch the CSVs.
Writes results/overleaf/sthereo_valley_ORB_CLAHEON_NORM-CLAHE_traj_tracked.png
and copies the existing diagnostics figure alongside it.
"""
import os, sys, shutil, importlib, numpy as np
sys.path.insert(0, 'src/mTIO')
from mTIO.evaluation import align_and_compute_metrics, plot_trajectory

NAME    = 'sthereo_valley_ORB_CLAHEON_NORM-CLAHE'
DIVERGE = 95.0          # hand-marked divergence (results/divergence_marks.csv)

cfg = importlib.import_module('mTIO.config_sthereo')
gt  = cfg.load_ground_truth()

raw  = np.loadtxt(f'results/{NAME}.txt')          # t x y z qx qy qz qw
traj4 = raw[:, :4]
print(f'traj samples: {len(traj4)}  gt samples: {len(gt["t"])}')

ours = align_and_compute_metrics(traj4, gt['t'], gt['p'],
                                 settle_time_s=None, divergence_time_s=DIVERGE)
print(f'track ATE [m] (0..{DIVERGE:.0f}s): {ours["pre_divergence_ate_rmse_m"]:.2f}')

title_info = {'run_id': NAME, 'dataset': 'sthereo', 'method': 'ORB',
              'clahe': 'on', 'normalize': 'NORM-CLAHE', 'mode': 'vio'}

os.makedirs('results/overleaf', exist_ok=True)
out = f'results/overleaf/{NAME}_traj_tracked.png'
plot_trajectory(traj4, gt['t'], gt['p'], ours, title_info, out,
                t_max=DIVERGE, gt_full_xyz=gt['p'])
print('saved ->', out)

# diagnostics figure already exists from the live run — copy it next to the traj
diag_src = f'results/{NAME}_diag.png'
diag_dst = f'results/overleaf/{NAME}_diag.png'
if os.path.exists(diag_src):
    shutil.copy(diag_src, diag_dst)
    print('copied ->', diag_dst)
else:
    print('WARN: diag not found:', diag_src)