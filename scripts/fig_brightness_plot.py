"""Quantitative brightness-constancy check on ROVTIO (Norm->CLAHE): mean 8-bit
intensity of the full frame and of a fixed central image patch over a window
around t=67 s. A physically slow-changing scene whose mapped intensity varies
frame-to-frame evidences the brightness-constancy violation.
Saved to results/overleaf/fig_brightness_plot.png.
"""
import os, sys, importlib, numpy as np
sys.path.insert(0, 'src/thermal_vo')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thermal_vo.common_params import PREPROC
import thermal_vo.dataloader2 as dl2

T0, T1 = 64.0, 70.0
cfg = importlib.import_module('thermal_vo.config_rovtio'); cal = cfg.load_camera_intrinsics()
L = dl2.ThermalDataLoader(cfg.CAM0_DIR, bit_depth=16, undistort=True,
                          K=cal['K'], D=cal['D'], distortion_model=cal['distortion_model'],
                          use_clahe=True, **PREPROC)
L.load_timestamps_file(cfg.CAM0_TS, unit='s')
tr = np.asarray(L.timestamps) - L.timestamps[0]
lo, hi = int(np.searchsorted(tr, T0)), int(np.searchsorted(tr, T1))

ts, full, patch = [], [], []
for i in range(lo, hi):
    img = L._load_single(L.image_paths[i]).astype(np.float32)
    h, w = img.shape
    p = img[h//2 - 40:h//2 + 40, w//2 - 40:w//2 + 40]   # fixed 80x80 central patch
    ts.append(tr[i]); full.append(img.mean()); patch.append(p.mean())
ts, full, patch = map(np.array, (ts, full, patch))

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(ts, full, 'k-', lw=1.3, label=f'full frame (std={full.std():.1f})')
ax.plot(ts, patch, 'b-', lw=1.5, label=f'fixed central patch (std={patch.std():.1f})')
ax.axvspan(67.13, 67.43, color='orange', alpha=0.15, label='frame strip window')
ax.set_xlabel('t [s]'); ax.set_ylabel('mean 8-bit intensity')
ax.set_title('ROVTIO (Norm$\\rightarrow$CLAHE): mapped intensity over time')
ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=9)
plt.tight_layout()
os.makedirs('results/overleaf', exist_ok=True)
out = 'results/overleaf/fig_brightness_plot.png'
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'full-frame std={full.std():.2f}  patch std={patch.std():.2f}  '
      f'patch range={patch.min():.1f}..{patch.max():.1f}')
print('saved ->', out)