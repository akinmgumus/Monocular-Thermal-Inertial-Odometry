"""Figure: pure inertial (IMU-only, no visual updates) dead-reckoning position
drift versus time, overlaid for each platform. Each IMU-only trajectory is
posyaw-aligned to GT over the first few seconds (so all curves start together),
then the absolute position error is plotted against time on a logarithmic axis
(the platforms differ by orders of magnitude). Characterises the RAW inertial
quality independently of the vision front-end.
Saved to results/overleaf/fig_imuonly.png.
"""
import os, sys, importlib, numpy as np
sys.path.insert(0, 'src/mTIO')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mTIO.evaluation import align_and_compute_metrics

ALIGN_S = 5.0      # posyaw fit window at the start (dead-reckoning anchor)
T_MAX   = 90.0     # focus on the drift onset (tracking windows are all < 120 s)

RUNS = [
    ('ROVTIO',     'config_rovtio',     'rovtio_alt1_ORB_CLAHEON_NORM-CLAHE_IMU',       'tab:blue'),
    ('STheReo',    'config_sthereo',    'sthereo_valley_ORB_CLAHEON_NORM-CLAHE_IMU',    'tab:green'),
    ('FIReStereo', 'config_firestereo', 'firestereo_frick1_ORB_CLAHEOFF_NORM-CLAHE_IMU','tab:red'),
    ('EuRoC',      'config_euroc',      'euroc_mh03_ORB_CLAHEOFF_IMU',                  '0.35'),
]

fig, ax = plt.subplots(figsize=(9, 4.6))
for label, cfgname, name, color in RUNS:
    cfg = importlib.import_module('mTIO.' + cfgname)
    gt  = cfg.load_ground_truth()
    raw = np.loadtxt(f'results/{name}.txt')
    traj4 = raw[:, :4]
    t = traj4[:, 0] - traj4[0, 0]
    m = align_and_compute_metrics(traj4, gt['t'], gt['p'],
                                  settle_time_s=None, divergence_time_s=ALIGN_S)
    err = np.asarray(m['_ate_series'])
    ls = '--' if label == 'EuRoC' else '-'
    ax.plot(t, err, ls, color=color, lw=1.8, label=label)
    print(f'{label:11s} drift@30s={err[min(np.searchsorted(t,30),len(err)-1)]:8.1f} m')

ax.set_yscale('log')
ax.set_xlim(0, T_MAX)
ax.set_ylim(0.05, 5e3)
ax.set_xlabel('t [s]  (from start, IMU-only dead-reckoning)')
ax.set_ylabel('position drift $\\|\\hat{p}-p_\\mathrm{gt}\\|$ [m]')
ax.set_title('Inertial-only dead-reckoning drift (no visual updates), '
             'posyaw-anchored over the first %g s' % ALIGN_S, fontsize=10)
ax.grid(alpha=0.3, which='both')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
os.makedirs('results/overleaf', exist_ok=True)
out = 'results/overleaf/fig_imuonly.png'
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print('saved ->', out)