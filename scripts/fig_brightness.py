"""Figure: four ROVTIO frames around t=36 s after Norm->CLAHE preprocessing,
showing the abrupt brightness change of the same surface (brightness-constancy
violation from per-frame percentile normalisation).
Saved to results/overleaf/fig_brightness.png.
"""
import os, sys, importlib, numpy as np
sys.path.insert(0, 'src/thermal_vo')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thermal_vo.common_params import PREPROC
import thermal_vo.dataloader2 as dl2     # Norm -> CLAHE

T_CENTER = 67.13        # seconds (relative to first frame)
STRIDE   = 1           # frames between the four shown (~0.12 s each @ 25 Hz)
N        = 10

cfg = importlib.import_module('thermal_vo.config_rovtio')
cal = cfg.load_camera_intrinsics()
L = dl2.ThermalDataLoader(cfg.CAM0_DIR, bit_depth=16, undistort=True,
                          K=cal['K'], D=cal['D'],
                          distortion_model=cal['distortion_model'],
                          use_clahe=True, **PREPROC)
L.load_timestamps_file(cfg.CAM0_TS, unit='s')
t_rel = np.asarray(L.timestamps) - L.timestamps[0]
i0 = int(np.searchsorted(t_rel, T_CENTER))
idxs = [i0 + k * STRIDE for k in range(N)]

fig, axes = plt.subplots(1, N, figsize=(3.6 * N, 3.6))
for ax, i in zip(axes, idxs):
    img = L._load_single(L.image_paths[i])
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'$t = {t_rel[i]:.2f}$~s', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
print('frames:', idxs, ' times:', [f'{t_rel[i]:.2f}' for i in idxs])
plt.tight_layout()
os.makedirs('results/overleaf', exist_ok=True)
out = 'results/overleaf/fig_brightness.png'
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print('saved ->', out)