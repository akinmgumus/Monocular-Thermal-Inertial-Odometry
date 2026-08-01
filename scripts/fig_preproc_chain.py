"""Figure: the thermal preprocessing chain applied to one real frame (ROVTIO),
in the adopted Norm->CLAHE order. Three panels: (1) raw radiometric 16-bit frame
shown with a min-max stretch, (2) after per-frame percentile (2-98%) 8-bit
normalisation, (3) after CLAHE (clip 3.0, 8x8). Panels 2-3 come straight from the
dataloader (use_clahe off/on); the raw panel is undistorted with the SAME map so
all three are geometrically aligned. Illustrates what each operator does to the
image, complementing the block-diagram pipelines in the Method chapter.
Saved to results/overleaf/fig_preproc_chain.png.
"""
import os, sys, importlib, numpy as np
sys.path.insert(0, 'src/mTIO')
import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mTIO.common_params import PREPROC
import mTIO.dataloader2 as dl2          # Norm -> CLAHE (adopted order)

cfg = importlib.import_module('mTIO.config_rovtio')
cal = cfg.load_camera_intrinsics()
kw  = dict(K=cal['K'], D=cal['D'], distortion_model=cal['distortion_model'])

L_clahe = dl2.ThermalDataLoader(cfg.CAM0_DIR, bit_depth=16, undistort=True,
                                use_clahe=True,  **kw, **PREPROC)
L_norm  = dl2.ThermalDataLoader(cfg.CAM0_DIR, bit_depth=16, undistort=True,
                                use_clahe=False, **kw, **PREPROC)
mid = len(L_clahe.image_paths) // 2

img_clahe = L_clahe._load_single(L_clahe.image_paths[mid])   # also builds undistort maps
img_norm  = L_norm._load_single(L_norm.image_paths[mid])

# raw radiometric with the SAME undistortion, shown on the FULL 16-bit scale
# (as the sensor delivers it): the ~3200-level values fill only a thin slice of
# 0..65535, so the frame reads as nearly flat/dark --- this is exactly why the
# 8-bit normalisation is needed.
raw = cv2.imread(L_clahe.image_paths[mid], cv2.IMREAD_UNCHANGED)
if L_clahe._undistort_maps is not None:
    raw = cv2.remap(raw, *L_clahe._undistort_maps, cv2.INTER_LINEAR)
rlo, rhi = float(raw.min()), float(raw.max())
print(f'raw range: {rlo:.0f}..{rhi:.0f} (16-bit)   frame {mid}/{len(L_clahe.image_paths)}')

RAW_FULL = 65535.0
panels = [
    (raw,       RAW_FULL, 'raw radiometric (16-bit, full scale)'),
    (img_norm,  255,      '8-bit normalised (percentile 2–98%)'),
    (img_clahe, 255,      '+ CLAHE (clip 3.0, $8{\\times}8$) — adopted'),
]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
for ax, (img, vmax, title) in zip(axes, panels):
    ax.imshow(img, cmap='gray', vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
# arrows between panels to convey the chain
for x in (0.345, 0.655):
    fig.text(x, 0.5, r'$\rightarrow$', ha='center', va='center', fontsize=20)
plt.tight_layout()
os.makedirs('results/overleaf', exist_ok=True)
out = 'results/overleaf/fig_preproc_chain.png'
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print('saved ->', out)