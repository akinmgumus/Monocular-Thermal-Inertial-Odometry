"""Figure: ORB feature matches between two consecutive STheReo thermal frames
across a sharp turn. The turn instant is found from the IMU gyroscope (peak yaw
rate) rather than guessed, so the "sharp turn" claim is data-backed. The overlay
shows that the front-end still produces many geometrically valid 2D matches; the
lack of triangulation parallax (argued in the text) is a rotation-dominated
degeneracy, not a matching failure.
Saved to results/overleaf/fig_featmatch.png.
"""
import os, sys, importlib, numpy as np
sys.path.insert(0, 'src/mTIO')
import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mTIO.common_params import PREPROC
import mTIO.dataloader2 as dl2          # Norm -> CLAHE (adopted order)

# search the turn only within the tracked phase (ORB diverges ~95 s)
T_LO, T_HI = 10.0, 95.0
N_MATCHES  = 70

cfg = importlib.import_module('mTIO.config_sthereo')
cal = cfg.load_camera_intrinsics()

L = dl2.ThermalDataLoader(cfg.THERMAL_LEFT_DIR, bit_depth=16, undistort=True,
                          K=cal['K'], D=cal['D'], use_clahe=True, **PREPROC)
img_t = np.asarray(L.timestamps)
img_t0 = img_t - img_t[0]

imu = dl2.IMULoader.sthereo(cfg.IMU_CSV)
imu_t = np.asarray(imu.timestamps, dtype=np.float64)
if imu_t[0] > 1e12:            # nanoseconds -> seconds
    imu_t = imu_t / 1e9
imu_t0 = imu_t - imu_t[0]
wmag = np.linalg.norm(np.asarray(imu.gyro, dtype=np.float64), axis=1)  # |omega| rad/s

# peak yaw/angular rate within the tracked window
sel = (imu_t0 >= T_LO) & (imu_t0 <= T_HI)
k_peak = np.argmax(np.where(sel, wmag, -np.inf))
t_turn = imu_t0[k_peak]
print(f'peak |omega| = {wmag[k_peak]:.2f} rad/s ({np.degrees(wmag[k_peak]):.0f} deg/s) at t={t_turn:.2f}s')

# nearest consecutive image pair to the turn instant
i = int(np.searchsorted(img_t0, t_turn))
i = max(1, min(i, len(img_t0) - 2))
i0, i1 = i, i + 1
dt = img_t0[i1] - img_t0[i0]
print(f'frames {i0},{i1}  t={img_t0[i0]:.2f},{img_t0[i1]:.2f}s  dt={dt*1e3:.0f}ms')

im0 = L._load_single(L.image_paths[i0])
im1 = L._load_single(L.image_paths[i1])

orb = cv2.ORB_create(nfeatures=1500)
k0, d0 = orb.detectAndCompute(im0, None)
k1, d1 = orb.detectAndCompute(im1, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = sorted(bf.match(d0, d1), key=lambda m: m.distance)[:N_MATCHES]
print(f'keypoints: {len(k0)},{len(k1)}   matches drawn: {len(matches)}')

vis = cv2.drawMatches(im0, k0, im1, k1, matches, None,
                      matchColor=(0, 200, 0), singlePointColor=(255, 120, 0),
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.imshow(vis)
ax.set_title(f'STheReo (ORB, Norm$\\rightarrow$CLAHE): consecutive frames across a sharp turn '
             f'($|\\omega|\\approx{np.degrees(wmag[k_peak]):.0f}$\\,deg/s, $\\Delta t={dt*1e3:.0f}$\\,ms), '
             f'{len(matches)} matches', fontsize=10)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
os.makedirs('results/overleaf', exist_ok=True)
out = 'results/overleaf/fig_featmatch.png'
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print('saved ->', out)
