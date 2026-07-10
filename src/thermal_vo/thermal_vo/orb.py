# ORB-based feature tracker (grid-distributed detection + descriptor matching).

import cv2
import numpy as np

class FeatureTrack:
    """
    Represents the lifecycle of a single feature point.

    MSCKF.update() reads two things from this object:
      - frame_ids: which cam_states it was observed in
      - keypoints: the pixel coordinate in each of those cam_states

    Nothing else is needed — pose, descriptor, ID and the like are of no use
    to the MSCKF. Only "which pixel, in which frame". The descriptor is kept
    here because the TRACKER itself uses it for matching (it never reaches
    the MSCKF).
    """
    def __init__(self, track_id, frame_id, kp, desc):
        self.track_id   = track_id
        self.frame_ids  = [frame_id]
        self.keypoints  = [kp]      # (x, y) tuple, in pixels
        self.descriptor = desc      # ORB BRIEF descriptor (32 bytes / 256 bits)



class ORBTracker:
    def __init__(self, n_features=800,
                 grid_rows=4,
                 grid_cols=4,
                 ratio_thresh=0.75,
                 ransac_thresh=2.0,
                 min_track_length=3,
                 fast_threshold=8,
                 edge_threshold=10,
                 max_pixel_displacement=40.0):

        self.n_features       = n_features
        self.grid_rows        = grid_rows
        self.grid_cols        = grid_cols
        self.ratio_thresh     = ratio_thresh
        self.ransac_thresh    = ransac_thresh
        # Minimum number of observations a track needs before it is retired
        # as "dead" for use in the MSCKF update.
        self.min_track_length = min_track_length

        # Maximum allowed inter-frame pixel displacement. On thermal data,
        # homogeneous regions can make the RANSAC F-matrix degenerate (8
        # spurious matches can settle into a mutually consistent but wrong
        # geometry and get counted as inliers). This acts as a motion prior
        # between the descriptor-level filter and RANSAC: matches whose
        # Euclidean distance between prev_kp and curr_kp exceeds this
        # threshold are dropped outright. At ~10 fps thermal, the platform's
        # typical inter-frame displacement is 5-20 px, so 40 px is a
        # reasonable upper bound. Pass None to disable the check.
        self.max_pixel_displacement = max_pixel_displacement

        # Target ORB features per grid cell, so they end up evenly
        # distributed across the image rather than clustered in
        # high-texture regions.
        self.n_per_cell = max(1, n_features // (grid_rows * grid_cols))

        self.orb = cv2.ORB_create(nfeatures=self.n_per_cell * 2,  # extract 2x the per-cell target, so the strongest survive ranking
                                  fastThreshold=fast_threshold,   # FAST corner-response threshold
                                  edgeThreshold=edge_threshold    # border margin excluded from detection
                                  )

        # ORB descriptors are binary, so a brute-force matcher with Hamming
        # distance is used. crossCheck is left False because the Lowe-ratio
        # test and the mutual-nearest-neighbour check are applied manually
        # below.
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.next_track_id = 0    # counter assigning a unique ID to each new track
        self.active_tracks = []   # points currently being tracked
        self.dead_tracks   = []   # tracks that lost their match this frame, fed to the MSCKF update

    def _extract_grid_orb(self, image):
        """Splits the image into a grid and keeps the strongest ORB features
        per cell, remapped to full-image coordinates. This spreads the
        detected features more evenly across the image than a single
        whole-image ORB call would."""
        h, w = image.shape
        all_kp, all_desc = [], []

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                y0 = r * h // self.grid_rows
                y1 = (r + 1) * h // self.grid_rows
                x0 = c * w // self.grid_cols
                x1 = (c + 1) * w // self.grid_cols
                cell = image[y0:y1, x0:x1]

                kp, desc = self.orb.detectAndCompute(cell, None)
                if desc is None or len(kp) == 0:
                    continue

                # Keep only the strongest keypoints per cell (by FAST response).
                if len(kp) > self.n_per_cell:
                    order = np.argsort([-k.response for k in kp])[:self.n_per_cell]
                    kp   = [kp[i] for i in order]
                    desc = desc[order]

                # Adjust keypoint coordinates to the full image frame of reference.
                for k in kp:
                    k.pt = (k.pt[0] + x0, k.pt[1] + y0)

                all_kp.extend(kp)
                all_desc.append(desc)

        if len(all_kp) == 0:
            return [], None

        all_desc = np.vstack(all_desc)
        return all_kp, all_desc

    def _match_descriptors(self, prev_desc, curr_desc):
        """
        Bidirectional, ratio-gated matching between the previous frame's
        track descriptors and the new frame's descriptors.

        Args:
            prev_desc: (N_prev, 32) uint8 — active_tracks' descriptors
            curr_desc: (N_curr, 32) uint8 — the new frame's ORB descriptors

        Returns:
            good_matches: list[cv2.DMatch], each a prev->curr correspondence
        """
        # At least 2 candidates are needed (k=2) to apply the Lowe-ratio test.
        if prev_desc is None or curr_desc is None or len(curr_desc) < 2 or len(prev_desc) < 2:
            return []

        # 1) Forward (prev -> curr, k=2): the Lowe-ratio test needs the top-2 NN.
        matches_fwd = self.matcher.knnMatch(prev_desc, curr_desc, k=2)

        # 2) Backward (curr -> prev, k=1): the mutual check only needs the top-1 NN.
        matches_bwd = self.matcher.knnMatch(curr_desc, prev_desc, k=1)

        # best_bwd[curr_idx] = nearest prev_idx. Dict for O(1) lookup.
        best_bwd = {}
        for pair in matches_bwd:
            if len(pair) >= 1:
                best_bwd[pair[0].queryIdx] = pair[0].trainIdx

        # 3) + 4) Lowe ratio + mutual nearest neighbour
        good_matches = []
        for pair in matches_fwd:
            if len(pair) != 2:
                continue
            m, n = pair  # m: 1-NN, n: 2-NN
            # m.queryIdx indexes prev_desc, m.trainIdx indexes curr_desc,
            # m.distance is the Hamming distance between the two descriptors.

            # Lowe ratio test: reject if the best match isn't decisively
            # closer than the second-best (an ambiguous match).
            if m.distance >= self.ratio_thresh * n.distance:
                continue

            # Mutual-NN check: does curr's nearest neighbour, looked up
            # backward, land back on the same prev index?
            if best_bwd.get(m.trainIdx) != m.queryIdx:
                continue

            good_matches.append(m)

        return good_matches

    # RANSAC
    def _RANSAC_outliers(self, good_matches, curr_kp):
        """
        RANSAC-based Fundamental-matrix estimation for geometric verification.

        Args:
            good_matches: list[cv2.DMatch], prev->curr matches (after the
                          Lowe-ratio + mutual-NN checks)
        Returns:
            inlier_mask: np.array(bool), same order as good_matches; True = inlier
        """
        if len(good_matches) < 8:  # the F-matrix estimate needs at least 8 correspondences
            return np.array([False] * len(good_matches))

        pts_prev = np.float32([self.active_tracks[m.queryIdx].keypoints[-1] for m in good_matches])
        pts_curr = np.float32([curr_kp[m.trainIdx].pt for m in good_matches])

        F, mask = cv2.findFundamentalMat(pts_prev, pts_curr, cv2.FM_RANSAC, self.ransac_thresh, 0.99)

        if mask is None:
            return np.array([False] * len(good_matches))
        return mask.ravel().astype(bool)

    def _retire_track(self, track):
        """
        What to do when a track drops out of tracking (dies). MSCKF.update()
        only cares about dead tracks satisfying min_track_length.

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

        Classic MSCKF "feature marginalization at pruning" behaviour.
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

    def process_frame(self, image, frame_id):
        """
        Called on every new frame. Updates the active track list, and places
        sufficiently long dead tracks into dead_tracks.

        Args:
            image:    the new frame (uint8 grayscale, preprocessed)
            frame_id: this frame's unique ID (same one used for the MSCKF cam_state)
        """
        # Reset dead_tracks at the start of every frame. MSCKF.update() only
        # cares about tracks retired THIS frame.
        self.dead_tracks = []

        # --- Step 1: extract this frame's features ---
        curr_kp, curr_desc = self._extract_grid_orb(image)

        # --- Edge case: no feature at all in this frame ---
        if curr_desc is None or len(curr_kp) == 0:
            for tr in self.active_tracks:
                self._retire_track(tr)
            self.active_tracks = []
            return

        # --- Edge case: first frame (no active track yet) ---
        if len(self.active_tracks) == 0:
            for kp, desc in zip(curr_kp, curr_desc):
                self.active_tracks.append(
                    FeatureTrack(self.next_track_id, frame_id, kp.pt, desc)
                )
                self.next_track_id += 1
            return

        # --- Step 2: gather previous descriptors from active_tracks ---
        prev_desc = np.array([t.descriptor for t in self.active_tracks])

        # --- Step 3: matching (Lowe ratio + mutual NN) ---
        good_matches = self._match_descriptors(prev_desc, curr_desc)
        n_after_desc = len(good_matches)

        # --- Step 3b: motion prior (max pixel displacement) ---
        # On thermal data, homogeneous regions can make the RANSAC F-matrix
        # degenerate; 8 spurious matches can settle into a mutually
        # consistent epipolar geometry and be counted as inliers. So, BEFORE
        # RANSAC, drop matches with a physically implausible large pixel jump.
        if self.max_pixel_displacement is not None and len(good_matches) > 0:
            max_d2 = self.max_pixel_displacement ** 2
            filtered = []
            for m in good_matches:
                p_prev = self.active_tracks[m.queryIdx].keypoints[-1]
                p_curr = curr_kp[m.trainIdx].pt
                dx = p_curr[0] - p_prev[0]
                dy = p_curr[1] - p_prev[1]
                if dx * dx + dy * dy <= max_d2:
                    filtered.append(m)
            good_matches = filtered
        n_after_disp = len(good_matches)

        # --- Step 4: discard geometric outliers via F-matrix RANSAC ---
        inlier_mask = self._RANSAC_outliers(good_matches, curr_kp)
        inlier_matches = [m for m, ok in zip(good_matches, inlier_mask) if ok]
        n_after_ransac = len(inlier_matches)

        # Diagnostics: how many matches survive at each stage?
        self.last_match_stats = {
            'after_desc':   n_after_desc,
            'after_disp':   n_after_disp,
            'after_ransac': n_after_ransac,
            'dropped_disp':   n_after_desc  - n_after_disp,
            'dropped_ransac': n_after_disp  - n_after_ransac,
        }

        # --- Step 5: three-way classification ---
        matched_prev = set()   # prev track indices that got a match
        matched_curr = set()   # curr keypoint indices that got a match
        new_active = []

        # SURVIVOR: extend existing tracks
        for m in inlier_matches:
            track = self.active_tracks[m.queryIdx]
            track.frame_ids.append(frame_id)
            track.keypoints.append(curr_kp[m.trainIdx].pt)
            # Descriptor refresh is ON: on thermal data the gradient shifts
            # quickly, so updating the descriptor to the latest appearance
            # improves matching stability. There is a drift risk, but the
            # RANSAC + Lowe-ratio + mutual-NN chain already catches most bad
            # matches.
            track.descriptor = curr_desc[m.trainIdx]
            new_active.append(track)
            matched_prev.add(m.queryIdx)
            matched_curr.add(m.trainIdx)

        # LOST: retire unmatched active tracks
        for i, track in enumerate(self.active_tracks):
            if i not in matched_prev:
                self._retire_track(track)

        # NEW: spawn from unmatched curr keypoints
        for i, (kp, desc) in enumerate(zip(curr_kp, curr_desc)):
            if i not in matched_curr:
                new_active.append(
                    FeatureTrack(self.next_track_id, frame_id, kp.pt, desc)
                )
                self.next_track_id += 1

        self.active_tracks = new_active
