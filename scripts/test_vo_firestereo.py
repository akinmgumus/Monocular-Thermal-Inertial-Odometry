"""
Pure monocular VO test on the FIReStereo (hawkins_4_train) thermal sequence.

Same algorithm as test_vo_only.py but with FIReStereo's K/D, image directory,
external timestamps file, and TUM-format ground truth. Lets us check whether
the tracking + epipolar-geometry pipeline that misbehaves on SThereo behaves
better on a different thermal sequence (different scene, different camera,
different motion profile).

Run:
    python3 scripts/test_vo_firestereo.py
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/thermal_vo'))

from thermal_vo.dataloader          import ThermalDataLoader
from thermal_vo.orb                 import ORBTracker
from thermal_vo.klt                 import KLTTracker


# ── FIReStereo hawkins_4_train ────────────────────────────────────────────────
IMAGE_DIR = '/home/akin/firestereo/dataset/stereo_thermal_depth/hawkins_4/hawkins_4_train/img_left'
TS_FILE   = '/home/akin/firestereo/dataset/stereo_thermal_depth/hawkins_4/hawkins_4_train/timestamps.txt'
GT_FILE   = '/home/akin/firestereo/firestereo/reconstruction/hawkins_run4_car/traj.txt'

K = np.array([[406.33233,  0.0,        311.51175],
              [  0.0,      406.95367,  241.75863],
              [  0.0,        0.0,        1.0     ]], dtype=np.float64)
D = np.array([-0.34952316, 0.10382263, -0.00014740, -0.00018344, 0.0],
             dtype=np.float64)
WIDTH, HEIGHT = 640, 512


METHOD        = 'ORB'  # 'ORB' or 'KLT'
VIZ_EVERY     = 2
STEP_SCALE    = 1.0 # pure monocular: unit-norm translation per step
STATIONARY_PX = 1.0 # median pixel-flow below this → vehicle stationary, skip pose update


def make_tracker(method):
    """Front-end selector: returns a tracker that exposes the FeatureTrack
    interface (process_frame, active_tracks, dead_tracks)."""
    if method == 'ORB':
        return ORBTracker(
            n_features=500, grid_rows=4, grid_cols=4,
            ratio_thresh=0.5, ransac_thresh=1.5,
            min_track_length=4,
        )
    if method == 'KLT':
        return KLTTracker(
            n_features=500,
            ransac_thresh=1.5, min_track_length=5,
        )
    raise ValueError(f"Unknown METHOD={method!r}; expected 'ORB' or 'KLT'.")


def run():
    loader = ThermalDataLoader(
        IMAGE_DIR,
        bit_depth=16,
        undistort=True, K=K, D=D,
        gaussian_sigma=1.0,
        gaussian_ksize=(5, 5),
    )
    loader.load_timestamps_file(TS_FILE, unit='ms')
    print(f"images: {len(loader)}")

    tracker = make_tracker(METHOD)
    print(f"front-end: {METHOD}")

    # GT in TUM format: timestamp x y z qx qy qz qw
    gt = np.loadtxt(GT_FILE, comments='#')
    gt_t   = gt[:, 0]
    gt_xyz = gt[:, 1:4]

    img_t0 = loader.timestamps[0]
    img_tN = loader.timestamps[-1]
    print(f"image t-range:  {img_t0:.3f} .. {img_tN:.3f}  (Δ={img_tN-img_t0:.1f}s)")
    print(f"GT    t-range:  {gt_t[0]:.3f} .. {gt_t[-1]:.3f}  (Δ={gt_t[-1]-gt_t[0]:.1f}s)")
    print(f"GT    x-range:  {gt_xyz[:,0].min():+.2f} .. {gt_xyz[:,0].max():+.2f}")
    print(f"GT    y-range:  {gt_xyz[:,1].min():+.2f} .. {gt_xyz[:,1].max():+.2f}")

    R_wc = np.eye(3)
    t_wc = np.zeros(3)
    traj = []

    plt.ion()
    fig, (ax_img, ax_traj) = plt.subplots(
        2, 1, figsize=(13, 11),
        gridspec_kw={'height_ratios': [1, 2.0]},
    )
    fig.suptitle(f"Pure monocular VO ({METHOD}) — FIReStereo hawkins_4_train", fontsize=12)

    ax_img.set_xticks([]); ax_img.set_yticks([])
    img_artist = ax_img.imshow(
        np.zeros((HEIGHT, 2 * WIDTH), dtype=np.uint8),
        cmap='gray', vmin=0, vmax=255,
    )
    match_lines = []
    text_artist = ax_img.text(
        5, 18, '', color='yellow', fontsize=10,
        bbox=dict(facecolor='black', alpha=0.5, pad=2),
    )

    # Pre-shift GT so it starts at (0,0) — VO also starts at origin.
    gt_xy_shift = gt_xyz[:, :2] - gt_xyz[0, :2]
    gt_line, = ax_traj.plot(gt_xy_shift[:, 0], gt_xy_shift[:, 1],
                            'r--', lw=1.4, label='GT (TUM, full)')
    vo_line, = ax_traj.plot([], [], 'b-',  lw=1.4, label='Pure VO')
    vo_pt    = ax_traj.scatter([], [], c='b', s=70, zorder=5, edgecolor='k')
    ax_traj.set_xlabel('x [m]'); ax_traj.set_ylabel('y [m]')
    ax_traj.set_aspect('equal')
    ax_traj.grid(alpha=0.3); ax_traj.legend(loc='best')

    # Lock axes to GT range so VO blowups can't squash the GT line out of view.
    pad = 5.0
    ax_traj.set_xlim(gt_xy_shift[:, 0].min() - pad, gt_xy_shift[:, 0].max() + pad)
    ax_traj.set_ylim(gt_xy_shift[:, 1].min() - pad, gt_xy_shift[:, 1].max() + pad)
    ax_traj.set_autoscale_on(False)

    prev_img     = None
    n_frame      = 0
    n_drops      = 0
    n_stationary = 0

    try:
        for frame_id in range(len(loader)):
            img, t = loader[frame_id]

            tracker.process_frame(img, frame_id)

            pts_prev, pts_curr = [], []
            for tr in tracker.active_tracks:
                if len(tr.keypoints) >= 2 and tr.frame_ids[-1] == frame_id and tr.frame_ids[-2] == frame_id - 1:
                    pts_prev.append(tr.keypoints[-2])
                    pts_curr.append(tr.keypoints[-1])

            valid_pose = False
            stationary = False
            R_rel = np.eye(3)
            t_rel = np.zeros(3)

            if prev_img is not None and len(pts_prev) >= 8:
                p1 = np.float32(pts_prev)
                p2 = np.float32(pts_curr)

                median_flow = float(np.median(np.linalg.norm(p2 - p1, axis=1)))
                if median_flow < STATIONARY_PX:
                    stationary = True
                    n_stationary += 1
                else:
                    E, mask = cv2.findEssentialMat(
                        p1, p2, K,
                        method=cv2.RANSAC, threshold=1.0, prob=0.999,
                    )
                    if E is not None and E.shape == (3, 3):
                        _, R_rel, t_rel_unit, _ = cv2.recoverPose(E, p1, p2, K, mask=mask)
                        t_rel = t_rel_unit.ravel() * STEP_SCALE
                        valid_pose = True

            if valid_pose:
                R_inv = R_rel.T
                t_inv = -R_inv @ t_rel
                t_wc  = t_wc + R_wc @ t_inv
                R_wc  = R_wc @ R_inv
            elif not stationary:
                n_drops += 1

            traj.append((t, *t_wc))
            n_frame += 1

            if n_frame % VIZ_EVERY == 0 and prev_img is not None:
                combined = np.hstack([prev_img, img])
                img_artist.set_data(combined)
                w = prev_img.shape[1]

                for ml in match_lines:
                    ml.remove()
                match_lines.clear()

                for p, q in zip(pts_prev, pts_curr):
                    line, = ax_img.plot(
                        [p[0], q[0] + w], [p[1], q[1]],
                        'g-', lw=0.4, alpha=0.55,
                    )
                    match_lines.append(line)

                text_artist.set_text(
                    f"frame {n_frame:4d}  pairs: {len(pts_prev):3d}  "
                    f"{'STATIONARY' if stationary else 'moving'}  "
                    f"stationary: {n_stationary}  pose_drops: {n_drops}"
                )

                traj_arr = np.array(traj)
                vo_line.set_data(traj_arr[:, 1], traj_arr[:, 2])
                vo_pt.set_offsets([[traj_arr[-1, 1], traj_arr[-1, 2]]])

                # GT artık sabit (tüm trajectory baştan çizildi). Burada bir
                # şey yapmaya gerek yok; sadece VO line/point güncellenir.
                ax_traj.set_title(
                    f"VO pos = ({t_wc[0]:+.2f}, {t_wc[1]:+.2f}, {t_wc[2]:+.2f})  "
                    f"step_scale={STEP_SCALE}"
                )
                plt.pause(0.001)

            prev_img = img

    except KeyboardInterrupt:
        print("\n[interrupted]")

    traj = np.array(traj)
    print(f"\nframes: {n_frame},  stationary: {n_stationary},  pose_drops: {n_drops}")
    if len(traj):
        print(f"final pos: ({traj[-1, 1]:+.3f}, {traj[-1, 2]:+.3f}, {traj[-1, 3]:+.3f})")
        os.makedirs('results', exist_ok=True)
        np.savetxt('results/vo_only_trajectory_firestereo.txt', traj, header='t x y z')
        plt.savefig('results/vo_only_final_firestereo.png', dpi=120)
        print("saved → results/vo_only_trajectory_firestereo.txt")
        print("       results/vo_only_final_firestereo.png")

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run()
