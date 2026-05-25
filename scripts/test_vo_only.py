"""
Pure monocular VO test (no IMU, no MSCKF).

Isolates the tracking + epipolar-geometry path so we can tell whether the
visual front-end is the source of trajectory error in the VIO pipeline.

Per cam frame:
    1. Detect ORB on the current undistorted+CLAHE thermal image (via FeatureTracker).
    2. Match active tracks against the previous frame (ratio + RANSAC inside the tracker).
    3. From the matched normalized-point pairs (prev, curr), recover relative
       camera motion with cv2.recoverPose → R_rel, t_rel.
    4. Integrate as T_world_cam_k = T_world_cam_{k-1} · T_{k-1,k}.

Notes on scale:
    Pure monocular VO has no metric scale — t_rel comes back unit-norm.
    We multiply t_rel by `STEP_SCALE` (default 1.0) so the trajectory's
    shape can be compared to GT independent of absolute distance. Sim3
    alignment (Umeyama) is the proper way to compare — left as TODO.

Run:
    python3 scripts/test_vo_only.py
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
from thermal_vo                     import config_sthereo as config


METHOD        = 'KLT'  # 'ORB' or 'KLT'
VIZ_EVERY     = 50     # redraw every Nth cam frame
STEP_SCALE    = 1.0   # scale factor for the trajectory (pure monocular VO has no metric scale, so this is just for visualization)
STATIONARY_PX = 1.0   # median pixel-flow below this → vehicle stationary, skip pose


def make_tracker(method):
    """Front-end selector: returns a tracker that exposes the FeatureTrack
    interface (process_frame, active_tracks, dead_tracks)."""
    if method == 'ORB':
        return ORBTracker(
            n_features=1000,
            grid_rows=4,
            grid_cols=4,
            ratio_thresh=0.7,
            ransac_thresh=2.0,
            min_track_length=4,
        )
    if method == 'KLT':
        return KLTTracker(
            n_features=1000,
            ransac_thresh=2.0, # KLT has more matches but also more outliers, so looser RANSAC thresh; tune as needed.
            min_track_length=3, # KLT tracks are more fragile, so retire to "dead" sooner for MSCKF update; tune as needed.
        )
    raise ValueError(f"Unknown METHOD={method!r}; expected 'ORB' or 'KLT'.")


def run():
    cal = config.load_camera_intrinsics()
    K, D = cal['K'], cal['D']

    loader = ThermalDataLoader(
        config.THERMAL_LEFT_DIR,
        bit_depth=16,
        undistort=True, K=K, D=D,
        gaussian_sigma=1.0,           # denoise thermal speckle before CLAHE
        gaussian_ksize=(5, 5),
    )
    print(f"images: {len(loader)}")

    tracker = make_tracker(METHOD)
    print(f"front-end: {METHOD}")

    # GT for shape comparison
    gt   = np.loadtxt(config.GT_LOCAL_POSE, delimiter=',')
    gt_t = gt[:, 0]; gt_xyz = gt[:, 1:4]

    # Pose: cam-in-world. Start at identity.
    R_wc = np.eye(3)
    t_wc = np.zeros(3)
    traj = []

    plt.ion()
    fig, (ax_img, ax_traj) = plt.subplots(
        2, 1, figsize=(13, 11),
        gridspec_kw={'height_ratios': [1, 2.0]},
    )
    fig.suptitle(f"Pure monocular VO ({METHOD}) — tracking + recoverPose only", fontsize=12)

    ax_img.set_xticks([]); ax_img.set_yticks([])
    img_artist = ax_img.imshow(
        np.zeros((cal['height'], 2 * cal['width']), dtype=np.uint8),
        cmap='gray', vmin=0, vmax=255,
    )
    match_lines = []
    text_artist = ax_img.text(
        5, 18, '', color='yellow', fontsize=10,
        bbox=dict(facecolor='black', alpha=0.5, pad=2),
    )

    gt_line,  = ax_traj.plot([], [], 'r--', lw=1.4, label='GT (local pose)')
    vo_line,  = ax_traj.plot([], [], 'b-',  lw=1.4, label='Pure VO')
    vo_pt     = ax_traj.scatter([], [], c='b', s=70, zorder=5, edgecolor='k')
    ax_traj.set_xlabel('x [m]'); ax_traj.set_ylabel('y [m]')
    ax_traj.set_aspect('equal')
    ax_traj.grid(alpha=0.3); ax_traj.legend(loc='best')

    prev_img    = None
    prev_kp_lookup = None   # frame_id of previous cam frame
    n_frame     = 0
    gt_origin   = None
    n_drops     = 0
    n_stationary = 0

    try:
        for frame_id in range(len(loader)):
            img, t = loader[frame_id]

            tracker.process_frame(img, frame_id)

            # Build (prev_pt, curr_pt) pairs from tracks observed in BOTH the
            # previous frame and the current one.
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

                # Stationary detector: pure monocular VO returns a unit-norm t̂
                # even when the camera is still — those random directions
                # accumulate into ~1m drift per held frame. Bail out when the
                # median flow tells us nothing is moving.
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
                # cv2.recoverPose returns motion from camera 1 to camera 2:
                # X_2 = R_rel · X_1 + t_rel.   So  T_world_cam2 = T_world_cam1 · T_12^{-1}.
                #   T_12^{-1}: R = R_rel^T, t = -R_rel^T · t_rel
                R_inv = R_rel.T
                t_inv = -R_inv @ t_rel

                t_wc = t_wc + R_wc @ t_inv
                R_wc = R_wc @ R_inv
            else:
                n_drops += 1

            traj.append((t, *t_wc))

            if gt_origin is None:
                gt_origin = np.array([
                    np.interp(t, gt_t, gt_xyz[:, 0]),
                    np.interp(t, gt_t, gt_xyz[:, 1]),
                    np.interp(t, gt_t, gt_xyz[:, 2]),
                ])

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

                gt_mask = gt_t <= t
                gt_xy   = gt_xyz[gt_mask, :2] - gt_origin[:2]
                gt_line.set_data(gt_xy[:, 0], gt_xy[:, 1])

                ax_traj.relim(); ax_traj.autoscale_view()
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
        np.savetxt('results/vo_only_trajectory.txt', traj, header='t x y z')
        plt.savefig('results/vo_only_final.png', dpi=120)
        print("saved → results/vo_only_trajectory.txt")
        print("       results/vo_only_final.png")

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run()
