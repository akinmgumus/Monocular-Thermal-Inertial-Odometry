"""Figure: same middle thermal frame processed with CLAHE->Norm (dataloader)
and Norm->CLAHE (dataloader2), for each of the three thermal datasets.
Saved to results/overleaf/fig_preproc_order.png.
"""
import os, sys, importlib, numpy as np
sys.path.insert(0, 'src/mTIO')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mTIO.common_params import PREPROC
import mTIO.dataloader  as dl1     # CLAHE -> Norm
import mTIO.dataloader2 as dl2     # Norm  -> CLAHE

def setup(ds):
    cfg = importlib.import_module(f'mTIO.config_{ds}')
    if ds == 'rovtio':
        cal = cfg.load_camera_intrinsics()
        cam = cfg.CAM0_DIR
        kw  = dict(K=cal['K'], D=cal['D'], distortion_model=cal['distortion_model'])
    elif ds == 'sthereo':
        cal = cfg.load_camera_intrinsics()
        cam = cfg.THERMAL_LEFT_DIR
        kw  = dict(K=cal['K'], D=cal['D'])
    elif ds == 'firestereo':
        cal = cfg.load_camera_intrinsics('cam0')
        cam = os.path.join(cfg.DATA_ROOT, 'processed', 'frick_1', 'cam0', 'data')
        kw  = dict(K=cal['K'], D=cal['D'])
    return cam, kw

DATASETS = [('rovtio', 'ROVTIO'), ('sthereo', 'STheReo'), ('firestereo', 'FIReStereo')]
fig, axes = plt.subplots(len(DATASETS), 2, figsize=(8, 4.2 * len(DATASETS)))

for row, (ds, label) in enumerate(DATASETS):
    try:
        cam, kw = setup(ds)
        L_cn = dl1.ThermalDataLoader(cam, bit_depth=16, undistort=True,
                                     use_clahe=True, **kw, **PREPROC)   # CLAHE->Norm
        L_nc = dl2.ThermalDataLoader(cam, bit_depth=16, undistort=True,
                                     use_clahe=True, **kw, **PREPROC)   # Norm->CLAHE
        mid = len(L_cn.image_paths) // 2
        img_cn = L_cn._load_single(L_cn.image_paths[mid])
        img_nc = L_nc._load_single(L_nc.image_paths[mid])
        for ax, img, sub in ((axes[row, 0], img_cn, r'CLAHE$\rightarrow$Norm'),
                             (axes[row, 1], img_nc, r'Norm$\rightarrow$CLAHE')):
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'{label}: {sub}', fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        print(f'{ds}: frame {mid}/{len(L_cn.image_paths)}  ok')
    except Exception as ex:
        for c in (0, 1):
            axes[row, c].text(0.5, 0.5, f'{label}: {ex}', ha='center', va='center',
                              fontsize=8, wrap=True, transform=axes[row, c].transAxes)
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
        print(f'{ds}: FAIL {ex}')

plt.tight_layout()
os.makedirs('results/overleaf', exist_ok=True)
out = 'results/overleaf/fig_preproc_order.png'
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print('saved ->', out)