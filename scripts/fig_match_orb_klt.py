"""Presentation figure: feature matches between the SAME pair of consecutive
STheReo thermal frames, once for ORB and once for KLT, so the two front-ends
can be compared directly. Uses each front-end's actual tuned parameters
(common_params.py). The frame pair is chosen at the run's sharpest turn
(peak gyro |omega|, IMU-derived so the choice is data-backed, not guessed) —
a calm mid-run instant was checked first and showed comparable match counts
for both front-ends (ORB 674 vs KLT 536), so it does not illustrate a real
difference; the turn is where the two front-ends' actual behaviour diverges.
Saves two files: results/overleaf/fig_match_orb.png and fig_match_klt.png.
"""
import os, sys, importlib, numpy as np
sys.path.insert(0, 'src/mTIO')
import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mTIO.common_params import PREPROC, ORB_PARAMS, KLT_PARAMS
import mTIO.dataloader2 as dl2          # Norm -> CLAHE (adopted order)

T_LO, T_HI = 10.0, 95.0     # search window: within ORB's tracked phase (diverges ~95s)

cfg = importlib.import_module('mTIO.config_sthereo')
cal = cfg.load_camera_intrinsics()

L = dl2.ThermalDataLoader(cfg.THERMAL_LEFT_DIR, bit_depth=16, undistort=True,
                          K=cal['K'], D=cal['D'], use_clahe=True, **PREPROC)
img_t0 = np.asarray(L.timestamps) - L.timestamps[0]

imu = dl2.IMULoader.sthereo(cfg.IMU_CSV)
imu_t = np.asarray(imu.timestamps, dtype=np.float64)
if imu_t[0] > 1e12:
    imu_t = imu_t / 1e9
imu_t0 = imu_t - imu_t[0]
wmag = np.linalg.norm(np.asarray(imu.gyro, dtype=np.float64), axis=1)
sel = (imu_t0 >= T_LO) & (imu_t0 <= T_HI)
k_peak = np.argmax(np.where(sel, wmag, -np.inf))
t_turn = imu_t0[k_peak]
print(f'peak |omega| = {np.degrees(wmag[k_peak]):.0f} deg/s at t={t_turn:.2f}s')

i0 = int(np.searchsorted(img_t0, t_turn))
i0 = max(0, min(i0, len(img_t0) - 2))
i1 = i0 + 1
dt = img_t0[i1] - img_t0[i0]
print(f'frames {i0},{i1}  t={img_t0[i0]:.2f},{img_t0[i1]:.2f}s  dt={dt*1e3:.0f}ms')

im0 = L._load_single(L.image_paths[i0])
im1 = L._load_single(L.image_paths[i1])


def side_by_side(im0, im1, pts0, pts1, title, out_path):
    h, w = im0.shape
    canvas = np.zeros((h, 2 * w, 3), dtype=np.uint8)
    canvas[:, :w]  = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
    canvas[:, w:]  = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    for (x0, y0), (x1, y1) in zip(pts0, pts1):
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)) + w, int(round(y1)))
        cv2.line(canvas, p0, p1, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, (255, 120, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, (255, 120, 0), -1, cv2.LINE_AA)
    vis = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.imshow(vis)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved ->', out_path)


# ---------------------------------------------------------------- ORB ----
orb = cv2.ORB_create(nfeatures=ORB_PARAMS['n_features'],
                     fastThreshold=ORB_PARAMS['fast_threshold'],
                     edgeThreshold=ORB_PARAMS['edge_threshold'])
k0, d0 = orb.detectAndCompute(im0, None)
k1, d1 = orb.detectAndCompute(im1, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
raw_matches = bf.match(d0, d1)

# Lowe-style displacement sanity gate (same spirit as ORBTracker's
# max_pixel_displacement) + keep the best-scoring matches for a clean plot.
max_d2 = ORB_PARAMS['max_pixel_displacement'] ** 2
good = []
for m in raw_matches:
    x0, y0 = k0[m.queryIdx].pt
    x1, y1 = k1[m.trainIdx].pt
    if (x1 - x0) ** 2 + (y1 - y0) ** 2 <= max_d2:
        good.append(m)
good = sorted(good, key=lambda m: m.distance)[:50]
pts0_orb = [k0[m.queryIdx].pt for m in good]
pts1_orb = [k1[m.trainIdx].pt for m in good]
print(f'ORB: keypoints {len(k0)},{len(k1)}  matches drawn: {len(good)}')

side_by_side(im0, im1, pts0_orb, pts1_orb,
            f'STheReo, ORB front-end: {len(good)} feature matches across two consecutive frames '
            f'($\\Delta t={dt*1e3:.0f}$ ms)',
            'results/overleaf/fig_match_orb.png')

# ---------------------------------------------------------------- KLT ----
corners0 = cv2.goodFeaturesToTrack(im0, maxCorners=KLT_PARAMS['n_features'],
                                   qualityLevel=KLT_PARAMS['quality_level'],
                                   minDistance=KLT_PARAMS['min_distance'])
corners0 = corners0.reshape(-1, 1, 2).astype(np.float32)

lk_params = dict(winSize=KLT_PARAMS['lk_win_size'], maxLevel=KLT_PARAMS['lk_max_level'],
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
next_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(im0, im1, corners0, None, **lk_params)
back_pts,  status_bwd, _ = cv2.calcOpticalFlowPyrLK(im1, im0, next_pts, None, **lk_params)

status_fwd = status_fwd.ravel().astype(bool)
status_bwd = status_bwd.ravel().astype(bool)
fb_err = np.linalg.norm(corners0.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1)
flow   = np.linalg.norm(next_pts.reshape(-1, 2) - corners0.reshape(-1, 2), axis=1)

keep = (status_fwd & status_bwd
        & (fb_err < KLT_PARAMS['fb_eps'])
        & (flow < KLT_PARAMS['max_pixel_displacement']))

idx_keep = np.where(keep)[0]
idx_best = idx_keep[np.argsort(fb_err[idx_keep])[:50]]
pts0_klt = corners0.reshape(-1, 2)[idx_best]
pts1_klt = next_pts.reshape(-1, 2)[idx_best]
print(f'KLT: corners {len(corners0)}  matches drawn: {len(pts0_klt)}')

side_by_side(im0, im1, pts0_klt, pts1_klt,
            f'STheReo, KLT front-end: {len(pts0_klt)} feature matches across two consecutive frames '
            f'($\\Delta t={dt*1e3:.0f}$ ms)',
            'results/overleaf/fig_match_klt.png')
