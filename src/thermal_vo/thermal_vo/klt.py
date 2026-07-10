# KLT (Kanade-Lucas-Tomasi) based feature tracker.
#
# Comparison with ORB (for thermal imagery):
#   + No descriptor — works directly on the intensity gradient. Avoids the
#     problem of BRIEF binary tests being unstable on thermal data.
#   + Sub-pixel accuracy comes naturally.
#   - The brightness-constancy assumption does not fully hold on thermal data
#     (temperature drifts over time); the FB-consistency check tries to catch
#     the resulting failures.
#   - A lost track cannot be re-acquired (no descriptor re-matching).
#     Continuously opening new tracks via refill is required.
#
# Public API kept identical to ORBTracker:
#   .process_frame(image, frame_id)
#   .active_tracks      : list[FeatureTrack]
#   .dead_tracks        : list[FeatureTrack]
#   .marginalize_at_prune(frame_id)
#   .last_match_stats   : dict (for diagnostics, updated every frame)
# This lets the MSCKF test script switch trackers with a one-line import change.


import cv2
import numpy as np


class FeatureTrack:
    """
    Represents the lifecycle of a single feature point.

    MSCKF.update() reads two things from this object:
      - frame_ids: which cam_states it was observed in
      - keypoints: the pixel coordinate in each of those cam_states

    Unlike the ORB version, there is NO `descriptor` field here — KLT does
    pixel-level optical flow, not descriptor matching.
    """
    def __init__(self, track_id, frame_id, kp):
        self.track_id   = track_id
        self.frame_ids  = [frame_id]
        self.keypoints  = [kp]      # (x, y) tuple, in pixels


class KLTTracker:
    def __init__(self,
                 n_features=1000,
                 fb_eps=1.5,
                 ransac_thresh=2.0,
                 min_track_length=3,
                 lk_win_size=(25, 25),
                 lk_max_level=3,
                 quality_level=0.01,
                 min_distance=8.0,
                 max_pixel_displacement=25.0,
                 refill_threshold=None):
        """
        Args:
            n_features: Maximum number of tracks to try to maintain at once.
                        This many Shi-Tomasi corners are targeted at bootstrap
                        and at refill.

            fb_eps:     Forward-backward round-trip tolerance (pixels).
                        When a track makes the frame1 -> frame2 -> frame1
                        round trip and returns farther than this distance
                        from where it started, it is dropped. 1.5 px is
                        strict; relaxed to 2.0 on thermal data if needed.

            ransac_thresh: F-matrix RANSAC inlier threshold (pixels). Catches
                        geometric outliers that passed the Lowe-ratio +
                        FB-consistency checks but violate the epipolar
                        geometry.

            min_track_length: Minimum number of observations a track needs
                        before it is placed in dead_tracks (i.e. fed to the
                        MSCKF update). Same as for ORB.

            lk_win_size: Lucas-Kanade window (pixels). The least-squares
                        solution assumes every pixel inside this window
                        "moves with the same (u,v) flow vector". A larger
                        window gives a stronger structure matrix but
                        increases edge artefacts. (25, 25) is a bit more
                        stable than (21, 21) for the smooth gradients typical
                        of thermal imagery.

            lk_max_level: Number of image pyramid levels (0 included).
                        max_level=3 means 4 levels total, the coarsest at
                        1/8 resolution. This lets ~30-40 px of inter-frame
                        motion be handled — critical during fast rotations.

            quality_level: Shi-Tomasi relative quality threshold. A corner is
                        accepted if its min(lambda_1, lambda_2) is at least
                        this fraction of the strongest corner's value.
                        Thermal imagery is low-contrast, so 0.005-0.01 is
                        typical; higher (0.05+) fails to find enough corners
                        on most thermal frames.

            min_distance: Minimum pixel distance between two Shi-Tomasi
                        corners (non-maximum-suppression radius). Enforces
                        spatial diversity and prevents clustering. The KLT
                        counterpart of ORB's grid logic.

            max_pixel_displacement: Maximum allowed inter-frame Euclidean
                        displacement of a track (pixels). Catches cases that
                        pass FB-consistency but where both the forward and
                        backward flow are wrong yet converge to a consistent
                        (incorrect) point — the same logic as the
                        motion-prior gate added on the ORB side.

            refill_threshold: When the active track count drops below this
                        threshold, new Shi-Tomasi corners are added via
                        refill up to n_features. None -> n_features // 2
                        (default).
        """
        self.n_features             = n_features
        self.fb_eps                 = fb_eps
        self.ransac_thresh          = ransac_thresh
        self.min_track_length       = min_track_length
        self.lk_win_size            = lk_win_size
        self.lk_max_level           = lk_max_level
        self.quality_level          = quality_level
        self.min_distance           = min_distance
        self.max_pixel_displacement = max_pixel_displacement
        self.refill_threshold       = (refill_threshold if refill_threshold is not None
                                       else n_features // 2)

        # Dict-form parameters for cv2.calcOpticalFlowPyrLK. Iteration
        # criterion: either 30 iterations, or a step delta < 0.01 pixel.
        self.lk_params = dict(
            winSize=lk_win_size,
            maxLevel=lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.next_track_id = 0    # Counter assigning a unique ID to each new track
        self.active_tracks = []   # Tracks currently being followed
        self.dead_tracks   = []   # Tracks retired this frame (fed to the MSCKF)
        self.prev_image    = None # Previous frame, used as the LK reference

        # For diagnostics — populated at the end of every process_frame.
        # The test script prints this every ~50 frames to see how much each
        # filtering stage is dropping.
        self.last_match_stats = {
            'tracked':      0,   # tracks LK managed to follow
            'after_fb':     0,   # survived FB-consistency
            'after_disp':   0,   # survived max_pixel_displacement
            'after_ransac': 0,   # survived the F-matrix RANSAC
        }


    # ------------------------------------------------------------------
    def _detect_corners(self, image, mask=None):
        """
        Shi-Tomasi corner detector. Finds pixels that are "good to track".

        What happens internally:
          1. Image gradient (Ix, Iy via Sobel).
          2. For each pixel, the 2x2 structure matrix
             M = [[sum Ix^2, sum Ix*Iy], [sum Ix*Iy, sum Iy^2]],
             summed over a small neighbourhood (cv2 default 3x3).
          3. Compute min(lambda_1, lambda_2) for each pixel; the higher ones
             count as corners: those with
             min(lambda_1, lambda_2) >= quality_level * max(min(lambda_i))_over_image.
          4. Apply non-maximum suppression with radius min_distance.
          5. Return at most n_features of the strongest corners.

        Args:
            image: uint8 grayscale (preprocessed thermal frame).
            mask:  optional uint8 mask; corners are NOT searched for where it
                   is 0. Used at refill time to mask out the area around
                   existing active tracks.

        Returns:
            (N, 2) float32 numpy array — each row is a corner's (x, y) pixel
            coordinate. N <= n_features. An empty (0, 2) array if no corner
            is found.
        """
        corners = cv2.goodFeaturesToTrack(
            image,
            maxCorners=self.n_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
        )
        # cv2 can return None (if it finds no corner at all in the image):
        # always return a 2D float32 array for uniform downstream behaviour.
        if corners is None:
            return np.empty((0, 2), dtype=np.float32)
        # cv2 gives shape (N, 1, 2) — flatten to (N, 2).
        return corners.reshape(-1, 2).astype(np.float32)


    # ------------------------------------------------------------------
    def _refill_mask(self, image_shape):
        """
        Builds the uint8 array that masks out the area around active tracks,
        so Shi-Tomasi knows where NOT to look for corners during refill.

        Mask logic:
          - Everything starts at 255 (= "search here").
          - A filled disc of radius `min_distance` is drawn around each
            active track's LAST keypoint, set to 0 (= "do not search").

        Choice of radius = min_distance: the same rule already in effect
        ("two Shi-Tomasi corners must be at least min_distance apart") is
        applied here between an existing track and a candidate new corner,
        for consistency.

        Args:
            image_shape: (H, W) tuple — image.shape[:2]
        Returns:
            (H, W) uint8 mask
        """
        mask = np.full(image_shape, 255, dtype=np.uint8)
        radius = int(round(self.min_distance))
        for tr in self.active_tracks:
            x, y = tr.keypoints[-1]
            cv2.circle(mask, (int(round(x)), int(round(y))), radius, 0, thickness=-1)
        return mask


    # ------------------------------------------------------------------
    def _retire_track(self, track):
        """
        What to do when a track drops out of tracking (dies).
        MSCKF.update() only cares about dead tracks that satisfy
        min_track_length — shorter tracks don't yield a good triangulation,
        so they are filtered out here from the start.

        Args:
            track: the FeatureTrack object to retire
        """
        if len(track.frame_ids) >= self.min_track_length:
            self.dead_tracks.append(track)

    def marginalize_at_prune(self, frame_id_to_drop):
        """
        Called when a cam_state is about to be dropped from the sliding
        window. Every ACTIVE track that observed that cam_state will lose
        its context in later frames (its old observations can no longer be
        tied to the MSCKF) — so they are moved to dead_tracks now, giving
        MSCKF.update() one last look at them. Otherwise, long,
        uninterrupted-tracking features (the most informative ones) would
        disappear without ever reaching the filter.

        This is the classic MSCKF "feature marginalization at pruning"
        behaviour. Works independently of the tracker type (ORB/KLT).
        """
        keep_active = []
        for tr in self.active_tracks:
            if frame_id_to_drop in tr.frame_ids:
                if len(tr.frame_ids) >= self.min_track_length:
                    self.dead_tracks.append(tr)
                # Tracks below min_track_length are silently dropped.
            else:
                keep_active.append(tr)
        self.active_tracks = keep_active


    # ------------------------------------------------------------------
    def process_frame(self, image, frame_id):
        """
        Called on every new frame. Updates the active track list, retires
        any track that fails the FB-consistency + max_pixel_displacement +
        F-matrix RANSAC chain, and refills with new Shi-Tomasi corners if
        needed.

        Args:
            image:    the new frame (uint8 grayscale, preprocessed)
            frame_id: this frame's unique ID (same one used for the MSCKF cam_state)
        """
        # Reset dead_tracks at the start of every frame. MSCKF.update() only
        # cares about tracks retired THIS frame — earlier frames' have
        # already been processed.
        self.dead_tracks = []

        # Reset the diagnostic counters.
        self.last_match_stats = {
            'tracked':      0,
            'after_fb':     0,
            'after_disp':   0,
            'after_ransac': 0,
        }

        # ---- PHASE A: bootstrap ----
        # Two entry conditions:
        #   1) No previous frame yet (first call, prev_image is None).
        #   2) There was a previous frame, but every track has been lost
        #      (active is empty).
        # We also (re)bootstrap from scratch in the second case, so the
        # window doesn't stay empty since the refill logic alone wouldn't
        # kick in. There is no previous feature to track in this frame, so
        # we skip the LK phases entirely and return directly.
        if self.prev_image is None or len(self.active_tracks) == 0:
            corners = self._detect_corners(image)
            for c in corners:
                self.active_tracks.append(
                    FeatureTrack(self.next_track_id, frame_id,
                                 (float(c[0]), float(c[1])))
                )
                self.next_track_id += 1
            self.prev_image = image
            return

        # ---- PHASE B: Pyramidal LK forward ----
        # prev_pts = the active tracks' last keypoints in the previous frame.
        # cv2.calcOpticalFlowPyrLK finds where these points are in the new
        # frame. Shape (N, 1, 2) — the layout cv2 expects.
        prev_pts = np.array(
            [t.keypoints[-1] for t in self.active_tracks],
            dtype=np.float32
        ).reshape(-1, 1, 2)

        # next_pts: (N, 1, 2) — the estimated new positions
        # status_fwd: (N, 1) uint8 — 1 = LK tracked it, 0 = failed
        # err: (N, 1) float — residual (unused, we have our own FB check)
        next_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_image, image, prev_pts, None, **self.lk_params
        )

        # Flatten status_fwd to a 1D bool array; the following phases work
        # with boolean masks.
        status_fwd = status_fwd.ravel().astype(bool)
        self.last_match_stats['tracked'] = int(status_fwd.sum())

        # Intermediate state — kept on self for Phases C-F. Once all phases
        # are settled into a single process_frame call, these three
        # attributes could be pulled into local variables; left as instance
        # attributes for now to ease incremental smoke testing.
        self._lk_prev_pts = prev_pts.reshape(-1, 2)
        self._lk_next_pts = next_pts.reshape(-1, 2)
        self._lk_status   = status_fwd

        # Note: self.prev_image is NOT updated yet. Phase F will do that.
        # The tracks are also still at their old keypoints — Phase F updates them.

        # ---- PHASE C: Backward LK + Forward-Backward Consistency ----
        # Mechanism for catching asymmetric LK failures: track the forward
        # LK's next_pts back in the reverse direction (image -> prev_image);
        # the resulting back_pts SHOULD land close to the original prev_pts.
        # An aperture problem, occlusion, or brightness change prevents this
        # round trip from closing, and the track is dropped.
        back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            image, self.prev_image,
            self._lk_next_pts.reshape(-1, 1, 2),
            None,
            **self.lk_params
        )
        status_bwd = status_bwd.ravel().astype(bool)
        back_pts = back_pts.reshape(-1, 2)

        # Round-trip error: ||prev_pts - back_pts||. Should be sub-pixel for
        # reliable tracks.
        fb_err = np.linalg.norm(self._lk_prev_pts - back_pts, axis=1)

        # Triple AND: forward OK + backward OK + round-trip within tolerance
        keep_fb = self._lk_status & status_bwd & (fb_err < self.fb_eps)

        self.last_match_stats['after_fb'] = int(keep_fb.sum())
        self._lk_keep_fb = keep_fb   # stashed for Phases D-F

        # ---- PHASE D: in-bounds + max_pixel_displacement ----
        # Two simple gates:
        #   (i)  Is the LK next_pt inside the image? Points that leave the
        #        bounds corrupt the next cv2 call and pollute the F-matrix
        #        RANSAC.
        #   (ii) Discard large flows that passed FB but are "physically
        #        implausible" (a symmetric LK failure — both forward and
        #        backward are wrong but converge to a consistent point).
        #        Same logic as the max_pixel_displacement gate on the ORB
        #        side.
        h, w = image.shape[:2]
        np_pts = self._lk_next_pts
        in_bounds = (np_pts[:, 0] >= 0) & (np_pts[:, 0] < w) & \
                    (np_pts[:, 1] >= 0) & (np_pts[:, 1] < h)

        flow = np.linalg.norm(np_pts - self._lk_prev_pts, axis=1)
        within_disp = flow < self.max_pixel_displacement

        keep_disp = self._lk_keep_fb & in_bounds & within_disp

        self.last_match_stats['after_disp'] = int(keep_disp.sum())
        self._lk_keep_disp = keep_disp

        # ---- PHASE E: F-matrix RANSAC ----
        # Last line of defence: discard tracks that look consistent in pixel
        # space but come from objects moving independently in 3D (a
        # pedestrian, a passing vehicle, a swaying branch) or from subtle LK
        # drift. Two-view geometry assumption: all matches come from the
        # same camera's ego-motion; those that violate it break the
        # epipolar constraint and RANSAC flags them as outliers.
        pts1 = self._lk_prev_pts[keep_disp]
        pts2 = self._lk_next_pts[keep_disp]

        if len(pts1) >= 8:   # 8 points minimum for the F-matrix estimate
            F, mask = cv2.findFundamentalMat(
                pts1, pts2, cv2.FM_RANSAC,
                self.ransac_thresh, 0.99,
            )
            if F is not None and mask is not None:
                ransac_inliers = mask.ravel().astype(bool)
            else:
                # F estimation failed — treat everything as an outlier (safe side)
                ransac_inliers = np.zeros(len(pts1), dtype=bool)
        else:
            # Fewer than 8 points -> the F-matrix is undefined, RANSAC is rejected.
            ransac_inliers = np.zeros(len(pts1), dtype=bool)

        # Expand the RANSAC inlier mask (defined over the keep_disp subset)
        # back to the full N length. Each i with keep_disp[i]=True maps, in
        # order, to an entry of ransac_inliers.
        keep_ransac = np.zeros_like(keep_disp)
        keep_ransac[keep_disp] = ransac_inliers

        self.last_match_stats['after_ransac'] = int(keep_ransac.sum())
        self._lk_keep_ransac = keep_ransac

        # ---- PHASE F: survivors update + retire + refill + commit ----
        # No state has changed so far — Phases B-E only computed
        # intermediate arrays. The actual mutations happen here:
        #   1) Advance surviving tracks with their new keypoint + frame_id.
        #   2) Retire dropped tracks (goes to dead_tracks if it satisfies
        #      min_track_length).
        #   3) If the active count fell below refill_threshold, add new
        #      corners via Shi-Tomasi.
        #   4) Replace prev_image with the new frame.
        #   5) Delete the intermediate _lk_* attributes (clean state).
        new_active = []
        for i, tr in enumerate(self.active_tracks):
            if keep_ransac[i]:
                x, y = self._lk_next_pts[i]
                tr.frame_ids.append(frame_id)
                tr.keypoints.append((float(x), float(y)))
                new_active.append(tr)
            else:
                self._retire_track(tr)
        self.active_tracks = new_active

        # Refill — top up if we dropped below the threshold.
        if len(self.active_tracks) < self.refill_threshold:
            mask = self._refill_mask(image.shape[:2])
            new_corners = self._detect_corners(image, mask=mask)
            needed = self.n_features - len(self.active_tracks)
            for c in new_corners[:needed]:
                self.active_tracks.append(
                    FeatureTrack(self.next_track_id, frame_id,
                                 (float(c[0]), float(c[1])))
                )
                self.next_track_id += 1

        # Move prev_image forward to this frame — the reference for the next call.
        self.prev_image = image

        # Clean up the intermediate state (clean exit + guards against
        # misuse on the next call).
        del self._lk_prev_pts, self._lk_next_pts, self._lk_status
        del self._lk_keep_fb, self._lk_keep_disp, self._lk_keep_ransac
