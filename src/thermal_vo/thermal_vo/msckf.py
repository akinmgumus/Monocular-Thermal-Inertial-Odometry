import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.stats import chi2 as _chi2_dist


def skew(v):
    """Convert a 3-vector into its 3x3 skew-symmetric (cross-product) matrix."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ]
    ])

def triangulate_dlt(normalized_obs, cam_states):
    """
    Linear DLT triangulation via SVD.

    Stacks the cross-product constraint  u × (P · X) = 0  for every observation,
    yielding two rows per view, and solves A · X = 0 by taking the right
    singular vector with the smallest singular value.

    Args:
        normalized_obs: (N, 2) array of normalized image points (i.e. K^-1
                        applied to undistorted pixel coords — no intrinsics).
        cam_states:     list of N CamState objects whose `rot` is R_world_cam
                        and `pos` is the camera origin in the world frame.

    Returns:
        X_world: (3,) feature position in the world frame, or None if the
                 solution is at infinity (poor conditioning, e.g. zero parallax).

    Notes:
        DLT is fast but biased — it minimises algebraic error, not reprojection
        error. For MSCKF measurement update we usually feed this as the initial
        guess to a Gauss-Newton refinement in inverse-depth coordinates, which
        handles low-parallax tracks more gracefully.
    """
    n = len(cam_states)
    if n < 2:
        raise ValueError("triangulation requires at least 2 views")

    A = np.zeros((2 * n, 4))
    for i, (cs, u) in enumerate(zip(cam_states, normalized_obs)):
        # Projection matrix in normalized coordinates: P = [R_cw | t_cw]
        # where R_cw maps world points into the camera frame.
        R_cw = cs.rot.as_matrix().T
        t_cw = -R_cw @ cs.pos
        P    = np.hstack([R_cw, t_cw[:, None]])

        A[2 * i    ] = u[0] * P[2] - P[0]
        A[2 * i + 1] = u[1] * P[2] - P[1]

    _, _, Vt = np.linalg.svd(A)
    X_hom = Vt[-1]
    if abs(X_hom[3]) < 1e-12:
        return None
    return X_hom[:3] / X_hom[3]

class CamState:
    """Camera state for MSCKF sliding window."""
    def __init__(self, frame_id, rot, pos):
        self.frame_id = frame_id
        self.rot = rot      # current — gets updated each EKF step
        self.pos = pos
        # FEJ snapshot — frozen at augmentation, never modified
        self.rot_fej = rot
        self.pos_fej = pos



class MSCKF:
    """
    Multi-State Constraint Kalman Filter — IMU propagation backbone.

    Error-state ordering (15 dof):
        [ 0: 3]  δθ     attitude error
        [ 3: 6]  δb_g   gyro bias
        [ 6: 9]  δv     velocity
        [ 9:12]  δb_a   accel bias
        [12:15]  δp     position

    World frame is z-up: gravity = +9.81 ẑ, and a stationary IMU reads +g
    along its +z body axis. With this convention,  R·a_body − g  is zero at rest.
    """

    def __init__(self,
                 K=None,
                 D=None,
                 pixel_noise_std=1.5,    # px, 1-sigma feature observation noise
                 min_parallax_deg=1.0,   # min bearing-angle separation for valid track
                 min_depth=0.1,          # m, lower bound on landmark depth
                 max_depth=500.0,        # m, upper bound on landmark depth
                 gn_max_iter=5,          # Gauss-Newton iterations for inverse-depth refinement
                 chi2_alpha=0.95,        # confidence level for per-track Mahalanobis gate
                 init_att_std=0.02,      # rad,   ~1.1 deg
                 init_bg_std=0.01,       # rad/s
                 init_vel_std=0.05,      # m/s
                 init_ba_std=0.05,       # m/s^2
                 init_pos_std=0.001):    # m
        self.pixel_noise_std  = float(pixel_noise_std)
        self.min_parallax_rad = np.deg2rad(min_parallax_deg)
        self.min_depth        = float(min_depth)
        self.max_depth        = float(max_depth)
        self.gn_max_iter      = int(gn_max_iter)
        # Pre-compute chi-square thresholds for typical track DOF. After
        # null-space projection a track contributes (2M - 3) residual rows,
        # so for M ∈ [2, 20] DOF ranges 1..37.
        self._chi2_thresh = {
            df: float(_chi2_dist.ppf(chi2_alpha, df)) for df in range(1, 60)
        }
        # Camera intrinsics & distortion. If image-level undistortion is done in
        # the dataloader, pass D=None (or zeros) — pixels_to_normalized then just
        # applies K^-1. Otherwise D is the plumb-bob (k1,k2,p1,p2,k3) vector and
        # cv2.undistortPoints handles undistortion + normalization in one shot.
        self.K = None if K is None else np.asarray(K, dtype=np.float64)
        self.D = None if D is None else np.asarray(D, dtype=np.float64).ravel()
        # 1. NOMINAL STATE
        self.nominal_pos = np.zeros(3)
        self.nominal_vel = np.zeros(3)
        self.nominal_rot = R.from_quat([0, 0, 0, 1])   # scipy quat layout: [x, y, z, w]

        self.bg = np.zeros(3)
        self.ba = np.zeros(3)

        self.gravity = np.array([0.0, 0.0, 9.81])

        # 2. ERROR-STATE COVARIANCE (per-block initial uncertainty)
        self.P_matrix = np.diag(np.concatenate([
            np.full(3, init_att_std**2),
            np.full(3, init_bg_std**2),
            np.full(3, init_vel_std**2),
            np.full(3, init_ba_std**2),
            np.full(3, init_pos_std**2),
        ]))

        # 3. CONTINUOUS-TIME PROCESS NOISE (Xsens MTi-300 datasheet)
        # Density σ² has units variance / second, so Q_d = Q_c · dt during propagation.
        var_gyro = (0.0001745 * 1) ** 2     # gyro angular random walk (rad/s/√Hz)²
        var_acc  = (0.0005886 * 1) ** 2     # accel velocity random walk (m/s²/√Hz)²
        var_bg   = (0.0000048 * 1) ** 2     # gyro bias random walk (rad/s/√s)²
        var_ba   = (0.0001471 * 1) ** 2     # accel bias random walk (m/s²/√s)²

        self.Q_matrix = np.diag([
            var_gyro, var_gyro, var_gyro,
            var_bg,   var_bg,   var_bg,
            var_acc,  var_acc,  var_acc,
            var_ba,   var_ba,   var_ba,
            0.0, 0.0, 0.0,                # position has no driving noise
        ])
        
        # 4. CAMERA STATES (for Sliding Window MSCKF)
        self.cam_states = []

        

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------
    def initialize_from_static(self, gyro_samples, accel_samples):
        """
        Bootstrap state assuming the IMU is stationary during the given samples.

          • gyro bias    = mean(gyro)
          • gravity mag. = ||mean(accel)||  (sanity-check around 9.81)
          • initial roll/pitch derived by aligning mean(accel) with world +z
          • yaw set to 0 (unobservable from IMU alone)
        """
        gyro_samples  = np.asarray(gyro_samples)
        accel_samples = np.asarray(accel_samples)

        self.bg = gyro_samples.mean(axis=0)

        g_body = accel_samples.mean(axis=0)
        g_norm = np.linalg.norm(g_body)
        self.gravity = np.array([0.0, 0.0, g_norm])

        # Find rotation that maps g_body → world +z (i.e. R · g_body = g_world)
        g_b_unit = g_body / g_norm
        z_world  = np.array([0.0, 0.0, 1.0])
        axis     = np.cross(g_b_unit, z_world)
        axis_n   = np.linalg.norm(axis)

        if axis_n < 1e-8:
            self.nominal_rot = R.identity()
        else:
            angle = np.arctan2(axis_n, np.dot(g_b_unit, z_world))
            self.nominal_rot = R.from_rotvec(axis / axis_n * angle)

        self.nominal_pos = np.zeros(3)
        self.nominal_vel = np.zeros(3)
        self.ba          = np.zeros(3)

    # ------------------------------------------------------------------
    # PROPAGATION
    # ------------------------------------------------------------------
    def predict(self, dt, gyro_meas, acc_meas):
        """
        Propagate nominal state forward and inflate covariance with one IMU sample.
            gyro_meas: [wx, wy, wz] (rad/s)
            acc_meas:  [ax, ay, az] (m/s^2)
        """
        # ---- STEP 1: NOMINAL STATE INTEGRATION ----
        # Strip biases from raw measurements
        omega = gyro_meas - self.bg
        accel = acc_meas - self.ba

        # Attitude: body-frame angular velocity → right-multiplicative quaternion update
        dq      = R.from_rotvec(omega * dt)
        rot_new = self.nominal_rot * dq

        # Velocity / position kinematics in world frame
        rot_matrix  = self.nominal_rot.as_matrix()
        accel_world = rot_matrix @ accel

        vel_new = self.nominal_vel + (accel_world - self.gravity) * dt
        pos_new = (self.nominal_pos
                   + self.nominal_vel * dt
                   + 0.5 * (accel_world - self.gravity) * dt**2)

        # ---- STEP 2: ERROR-STATE TRANSITION MATRIX F ----
        # Continuous-time linearization of the error dynamics.
        F_matrix = np.zeros((15, 15))
        I_3 = np.eye(3)

        # Attitude error: dδθ/dt = -[ω×]δθ - δb_g
        F_matrix[0:3,   0:3]  = -skew(omega)
        F_matrix[0:3,   3:6]  = -I_3

        # Velocity error: dδv/dt = -R [a×] δθ - R δb_a
        F_matrix[6:9,   0:3]  = -rot_matrix @ skew(accel)
        F_matrix[6:9,   9:12] = -rot_matrix

        # Position error: dδp/dt = δv
        F_matrix[12:15, 6:9]  = I_3

        # ---- STEP 3: COVARIANCE PROPAGATION ----
        # First-order Euler discretization (sufficient at 100 Hz IMU rate)
        Phi        = np.eye(15) + F_matrix * dt
        Q_discrete = self.Q_matrix * dt

        # Full Φ is block_diag(Phi_imu, I_cam): cam states are static between
        # updates, but the IMU↔cam cross block must still propagate through
        # the IMU dynamics — skipping it freezes stale correlations and the
        # filter slowly diverges through inconsistent Kalman gains.
        P = self.P_matrix
        P_II_new = Phi @ P[:15, :15] @ Phi.T + Q_discrete
        if P.shape[0] > 15:
            P_IC_new = Phi @ P[:15, 15:]
            P[:15, 15:] = P_IC_new
            P[15:, :15] = P_IC_new.T
        P[:15, :15] = 0.5 * (P_II_new + P_II_new.T)
        self.P_matrix = P

        # ---- STEP 4: COMMIT NEW NOMINAL STATE ----
        self.nominal_rot = rot_new
        self.nominal_vel = vel_new
        self.nominal_pos = pos_new

        # ── TANI: P büyüme eğrisi ────────────────────────────────────────
        # Her 50 predict'te bir P'nin 1σ değerlerini bas. Amaç: ilk kamera
        # update'inden ÖNCE P nasıl büyüyor — düzgün/yavaş mı (P dürüst)
        # yoksa sıçrıyor mu (predict kovaryans bug'ı). Sorun bulununca sil.
        self._diag_predict_count = getattr(self, '_diag_predict_count', 0) + 1
        if self._diag_predict_count % 50 == 0:
            Pd = np.sqrt(np.clip(np.diag(self.P_matrix), 0.0, None))
            print(f"[predict {self._diag_predict_count:5d}] P 1σ: "
                  f"att={np.linalg.norm(Pd[0:3]):.4f}rad "
                  f"bg={np.linalg.norm(Pd[3:6]):.5f} "
                  f"vel={np.linalg.norm(Pd[6:9]):.4f}m/s "
                  f"ba={np.linalg.norm(Pd[9:12]):.5f} "
                  f"pos={np.linalg.norm(Pd[12:15]):.4f}m", flush=True)

    def augment_state(self, frame_id, R_imu_cam, t_imu_cam):
        """
        Append a new camera state to the sliding window and grow the covariance
        accordingly. Each camera state contributes 6 dof (3 attitude + 3 position).

        Args:
            frame_id:  identifier of the image attached to this clone
            R_imu_cam: scipy Rotation, rotation that maps camera vectors into the
                       IMU frame (i.e. v_imu = R_imu_cam · v_cam)
            t_imu_cam: 3-vector, position of the camera origin expressed in the
                       IMU frame

        Note: SThereo's `imu_2_thermal_left` file stores T_cam_imu (it maps IMU
        points → camera points). To call this method, invert it first:
            R_imu_cam = T_cam_imu[:3,:3].T
            t_imu_cam = -R_imu_cam @ T_cam_imu[:3,3]
        """
        R_world_imu = self.nominal_rot.as_matrix()

        # ---- STEP 1: COMPUTE THE CAMERA POSE IN THE WORLD FRAME ----
        # R_world_cam = R_world_imu · R_imu_cam
        rot_cam = self.nominal_rot * R_imu_cam

        # p_world_cam = p_world_imu + R_world_imu · t_imu_cam
        lever_arm_world = R_world_imu @ t_imu_cam
        pos_cam = self.nominal_pos + lever_arm_world

        self.cam_states.append(CamState(frame_id, rot_cam, pos_cam))

        # ---- STEP 2: BUILD THE AUGMENTATION JACOBIAN ----
        # J_imu (6x15): how the new camera's error state depends on the IMU error state.
        # All derivatives are taken with body-frame (right-multiplicative) attitude
        # error, matching the convention used in predict().
        J_imu = np.zeros((6, 15))

        # ∂δθ_cam / ∂δθ_imu = R_cam_imu  (body-frame attitude error transfer)
        J_imu[0:3, 0:3] = R_imu_cam.as_matrix().T

        # ∂δp_cam / ∂δp_imu = I
        J_imu[3:6, 12:15] = np.eye(3)

        # ∂δp_cam / ∂δθ_imu = -R_world_imu · [t_imu_cam]×   (lever-arm coupling)
        # A pure rotation of the IMU swings the camera around its lever arm.
        J_imu[3:6, 0:3] = -R_world_imu @ skew(t_imu_cam)

        # Old camera states do not influence the new one — pad J_full with zeros.
        current_size = self.P_matrix.shape[0]
        J_full = np.zeros((6, current_size))
        J_full[:, :15] = J_imu

        # ---- STEP 3: COVARIANCE AUGMENTATION ----
        # P_aug = [I; J] · P · [I; J]^T   →   block form below.
        P_11 = self.P_matrix
        P_12 = self.P_matrix @ J_full.T
        P_21 = J_full @ self.P_matrix
        P_22 = J_full @ self.P_matrix @ J_full.T

        self.P_matrix = np.block([
            [P_11, P_12],
            [P_21, P_22],
        ])

        # Symmetrize to absorb rounding errors after each growth.
        self.P_matrix = 0.5 * (self.P_matrix + self.P_matrix.T)

    # ------------------------------------------------------------------
    # MEASUREMENT-UPDATE HELPERS
    # ------------------------------------------------------------------
    def _cam_state_index(self, frame_id):
        """Linear lookup for the camera state with the given frame_id.
        Returns the list index (== block index in P), or None if not found.
        Sliding window is small (~15-20), so O(N) is fine."""
        for i, cs in enumerate(self.cam_states):
            if cs.frame_id == frame_id:
                return i
        return None

    # def _compute_track_jacobians_no_depth(self, track, p_f_world):
    #     """Same as _compute_track_jacobians but without min/max_depth check.

    #     FEJ: Jacobians linearized at the FROZEN first-estimate pose of each
    #     cam state (cs.rot_fej, cs.pos_fej), while the residual uses the
    #     CURRENT estimate. This keeps observability properties consistent —
    #     yaw and global position remain unobservable across the filter run,
    #     preventing the spurious-correction cascade that drove the divergence.
    #     """
    #     state_size = self.P_matrix.shape[0]
    #     valid = []
    #     for fid, kp in zip(track.frame_ids, track.keypoints):
    #         idx = self._cam_state_index(fid)
    #         if idx is not None:
    #             valid.append((idx, kp))
    #     if len(valid) < 2:
    #         return None

    #     pixels = np.array([kp for _, kp in valid], dtype=np.float64)
    #     z      = self.pixels_to_normalized(pixels)
    #     M = len(valid)
    #     H_x = np.zeros((2 * M, state_size))
    #     H_f = np.zeros((2 * M, 3))
    #     r   = np.zeros(2 * M)

    #     for k, ((idx, _), z_k) in enumerate(zip(valid, z)):
    #         cs = self.cam_states[idx]

    #         # ---- FEJ pose (Jacobian linearization) ----
    #         R_wc_fej    = cs.rot_fej.as_matrix()
    #         R_cw_fej    = R_wc_fej.T
    #         p_f_cam_fej = R_cw_fej @ (p_f_world - cs.pos_fej)
    #         X, Y, Z     = p_f_cam_fej

    #         if abs(Z) < 1e-3:   # only guard against division-by-zero
    #             return None

    #         inv_Z = 1.0 / Z
    #         J_proj = np.array([
    #             [inv_Z, 0.0,   -X * inv_Z**2],
    #             [0.0,   inv_Z, -Y * inv_Z**2],
    #         ])

    #         # Jacobian blocks use FEJ pose
    #         H_f[2*k:2*k+2, :] = J_proj @ R_cw_fej
    #         col0 = 15 + 6 * idx
    #         H_x[2*k:2*k+2, col0:col0+3]   = J_proj @ skew(p_f_cam_fej)
    #         H_x[2*k:2*k+2, col0+3:col0+6] = -J_proj @ R_cw_fej

    #         # ---- Current pose (residual) ----
    #         R_wc    = cs.rot.as_matrix()
    #         R_cw    = R_wc.T
    #         p_f_cam = R_cw @ (p_f_world - cs.pos)
    #         Xc, Yc, Zc = p_f_cam
    #         if abs(Zc) < 1e-3:
    #             return None
    #         z_hat = np.array([Xc / Zc, Yc / Zc])
    #         r[2*k:2*k+2] = z_k - z_hat

    #     return H_x, H_f, r


    def _compute_track_jacobians(self, track, p_f_world):
        """
        Build the per-track measurement Jacobian for one retired feature track.

        FEJ (First-Estimate Jacobians): the Jacobian blocks H_x and H_f are
        linearized at the FROZEN first-estimate pose of each cam state
        (cs.rot_fej, cs.pos_fej, captured at augmentation), while the
        residual z − z_hat is computed against the CURRENT estimate
        (cs.rot, cs.pos). This keeps observability properties of the
        nonlinear system consistent across the filter run — yaw and
        global position remain unobservable, which prevents the spurious
        correction loop that drives yaw drift in plain EKF-MSCKF.

        For the M' views that pass the depth gate (M' ≤ M observations still
        in the sliding window), returns:
            H_x: (2M', current_state_size) — derivative w.r.t. full error state
            H_f: (2M', 3)                  — derivative w.r.t. landmark position
            r  : (2M',)                    — residual z - h(x̂)

        Only the 6 columns corresponding to each observing cam state are nonzero
        in H_x; the IMU-state columns (0:15) and other cam-state columns stay 0.
        Observations whose frame_id is not in the current sliding window (e.g.
        already pruned) — or whose depth falls outside [min_depth, max_depth] in
        either the FEJ or the current pose — are skipped individually. The track
        is rejected outright (return None) only if fewer than 2 views survive.
        """
        state_size = self.P_matrix.shape[0]

        # First pass: keep only observations whose cam state is still in the window.
        valid = []
        for fid, kp in zip(track.frame_ids, track.keypoints):
            idx = self._cam_state_index(fid)
            if idx is not None:
                valid.append((idx, kp))

        if len(valid) < 2:
            return None  # need ≥2 views for a meaningful constraint

        pixels = np.array([kp for _, kp in valid], dtype=np.float64)
        z      = self.pixels_to_normalized(pixels)   # (M, 2)

        # ---- FEJ ANCHORING ----
        # p_f_world, GN tarafından kameraların GÜNCEL pozlarına göre üçgenlendi.
        # H_x/H_f ise FEJ (dondurulmuş) pozlarda lineerize ediliyor. Güncel
        # landmark'ı doğrudan FEJ pozlara iz düşürmek geometrik tutarsızlık
        # yaratır → yaw gözlemlenebilirliği bozulur → sahte attitude düzeltmesi.
        # Çözüm: landmark'ı çapa kameranın (valid[0]) güncel→FEJ rijit
        # dönüşümüyle FEJ dünyasına demirle, Jacobian'larda onu kullan.
        idx0          = valid[0][0]
        cs0           = self.cam_states[idx0]
        p_f_c0        = cs0.rot.as_matrix().T @ (p_f_world - cs0.pos)
        p_f_world_fej = cs0.rot_fej.as_matrix() @ p_f_c0 + cs0.pos_fej

        # Per-view rows are collected into lists, not a pre-sized (2M, ·) array:
        # a view whose depth (FEJ or current) falls outside [min_depth, max_depth]
        # is skipped INDIVIDUALLY instead of discarding the whole track. A long
        # track usually has one or two geometrically bad views while the rest
        # are fine — and long tracks are the most informative ones for the
        # filter, so dropping them wholesale starves the measurement update.
        H_x_rows, H_f_rows, r_rows = [], [], []

        for (idx, _), z_k in zip(valid, z):
            cs = self.cam_states[idx]

            # ---- FEJ pose: linearize Jacobian at the FROZEN first estimate ----
            R_wc_fej    = cs.rot_fej.as_matrix()
            R_cw_fej    = R_wc_fej.T
            p_f_cam_fej = R_cw_fej @ (p_f_world_fej - cs.pos_fej)
            X_f, Y_f, Z_f = p_f_cam_fej

            # ---- Current pose: used for the residual ----
            R_cw    = cs.rot.as_matrix().T
            p_f_cam = R_cw @ (p_f_world - cs.pos)
            Xc, Yc, Zc = p_f_cam

            # Depth gate — skip THIS view if either the FEJ Jacobian point or
            # the current residual point is behind the camera / absurdly far.
            if (Z_f < self.min_depth or Z_f > self.max_depth or
                    Zc < self.min_depth or Zc > self.max_depth):
                continue

            inv_Z_f = 1.0 / Z_f
            J_proj_fej = np.array([
                [inv_Z_f, 0.0,    -X_f * inv_Z_f**2],
                [0.0,     inv_Z_f, -Y_f * inv_Z_f**2],
            ])

            # H_f block: ∂h/∂p_f^G = J_proj · R_cw   (linearized at FEJ)
            H_f_k = J_proj_fej @ R_cw_fej

            # H_x block for this cam state at FEJ: 2x6 = J_proj · [ [p_f_cam]× , -R_cw ]
            H_x_k = np.zeros((2, state_size))
            col0 = 15 + 6 * idx
            H_x_k[:, col0:col0+3]   = J_proj_fej @ skew(p_f_cam_fej)  # ∂h/∂δθ
            H_x_k[:, col0+3:col0+6] = -J_proj_fej @ R_cw_fej          # ∂h/∂δp

            # Residual against the CURRENT estimate.
            z_hat = np.array([Xc / Zc, Yc / Zc])

            H_x_rows.append(H_x_k)
            H_f_rows.append(H_f_k)
            r_rows.append(z_k - z_hat)

        # Need ≥2 surviving views: null-space projection strips 3 dof from H_f,
        # so 2 views (4 rows) leave exactly one usable constraint row.
        if len(r_rows) < 2:
            return None

        return np.vstack(H_x_rows), np.vstack(H_f_rows), np.concatenate(r_rows)

    def _max_parallax(self, cam_states, normalized_obs):
        """
        Maximum bearing-angle separation across all observation pairs of a
        single track, measured in the world frame.

        For each view we lift the normalized observation [u, v] to a unit
        bearing ray [u, v, 1]/||·|| in the camera frame and rotate to world.
        The minimum cosine across all pairs gives the largest angular spread.

        A small parallax (< ~1°) means H_f is numerically rank-deficient and
        the Mourikis null-space projection produces phantom directions —
        gating with this prevents most divergence in slow / stationary
        segments without introducing any motion model assumption.
        """
        bearings = np.empty((len(cam_states), 3))
        for i, (cs, z) in enumerate(zip(cam_states, normalized_obs)):
            b = np.array([z[0], z[1], 1.0])
            b = b / np.linalg.norm(b)
            bearings[i] = cs.rot.as_matrix() @ b

        cos_M = np.clip(bearings @ bearings.T, -1.0, 1.0)
        np.fill_diagonal(cos_M, 1.0)
        return float(np.arccos(cos_M.min()))

    def _gauss_newton_refine(self, p_f_world, cam_states, normalized_obs):
        """
        Inverse-depth Gauss-Newton refinement of a triangulated landmark.

        Anchored to the first cam state in the track. Parametrize
            theta = (alpha, beta, rho)
            X_anchor = (alpha/rho, beta/rho, 1/rho)
        which is well-defined as rho → 0 (point at infinity), unlike
        Cartesian-space refinement.

        For each view i, the predicted bearing (up to a positive scale rho) is
            h_i = R_cam_world_i · R_world_anchor · [alpha, beta, 1]
                + rho · R_cam_world_i · (p_anchor - p_i)
        and the predicted normalized observation is z_hat_i = (h_i[0]/h_i[2],
        h_i[1]/h_i[2]) — note rho cancels in this ratio for a fixed ray
        through the anchor, so the parametrization stays smooth.

        Iterates GN  delta = (J^T J)^-1 J^T r, theta += delta, with early
        exit on small step. Returns the refined p_f in the world frame, or
        None if rho ends up negative / below min_depth / above max_depth or
        if the normal equations become singular.
        """
        anchor = cam_states[0]
        R_wa   = anchor.rot.as_matrix()
        p_a    = anchor.pos

        # Initial (alpha, beta, rho) from the DLT estimate, in anchor frame.
        X_a = R_wa.T @ (p_f_world - p_a)
        if X_a[2] < self.min_depth or X_a[2] > self.max_depth:
            return None
        alpha = X_a[0] / X_a[2]
        beta  = X_a[1] / X_a[2]
        rho   = 1.0 / X_a[2]

        M = len(cam_states)
        for _ in range(self.gn_max_iter):
            J_full = np.zeros((2 * M, 3))
            r_full = np.zeros(2 * M)

            for k, (cs, z) in enumerate(zip(cam_states, normalized_obs)):
                R_cw    = cs.rot.as_matrix().T
                p_diff  = p_a - cs.pos               # world-frame baseline
                R_total = R_cw @ R_wa                # anchor-cam → view-cam

                h = R_total @ np.array([alpha, beta, 1.0]) + rho * (R_cw @ p_diff)
                X, Y, Z = h
                if Z < 1e-3:
                    return None

                inv_Z  = 1.0 / Z
                z_hat  = np.array([X * inv_Z, Y * inv_Z])
                J_proj = np.array([
                    [inv_Z, 0.0,   -X * inv_Z**2],
                    [0.0,   inv_Z, -Y * inv_Z**2],
                ])
                # ∂h/∂(alpha, beta, rho)
                J_h = np.column_stack([R_total[:, 0], R_total[:, 1], R_cw @ p_diff])

                J_full[2*k:2*k+2, :] = J_proj @ J_h
                r_full[2*k:2*k+2]    = z - z_hat

            # GN normal equations: (J^T J) delta = J^T r ;  theta += delta.
            try:
                delta = np.linalg.solve(J_full.T @ J_full, J_full.T @ r_full)
            except np.linalg.LinAlgError:
                return None

            alpha += delta[0]
            beta  += delta[1]
            rho   += delta[2]

            if np.linalg.norm(delta) < 1e-7:
                break

        # Final sanity bounds on inverse depth.
        if rho <= 0.0:
            return None
        depth = 1.0 / rho
        if depth < self.min_depth or depth > self.max_depth:
            return None

        X_a_refined = np.array([alpha * depth, beta * depth, depth])
        return R_wa @ X_a_refined + p_a

    def _track_mahalanobis_sq(self, H_o, r_o, sigma_norm):
        """
        Squared Mahalanobis distance for one track's null-space-projected
        residual, used as a per-track outlier gate before stacking into the
        joint EKF update.

            d² = r_o^T (H_o P H_o^T + R_meas)^-1 r_o

        Compared against chi-square quantile at confidence chi2_alpha with
        df = len(r_o). Catches tracker outliers that survived RANSAC and
        any geometry corruption that GN refinement did not fully fix.
        """
        R_meas = (sigma_norm ** 2) * np.eye(H_o.shape[0])
        S = H_o @ self.P_matrix @ H_o.T + R_meas
        return float(r_o @ np.linalg.solve(S, r_o))

    def prune_cam_states(self, max_window_size):
        """
        FIFO sliding-window pruning. Drops the oldest camera state(s) until the
        window is at or below max_window_size, removing both the CamState entry
        and its 6 rows/cols from the covariance.

        Call AFTER update() — pruning before update would discard observations
        in the oldest cam state before they ever feed the filter.

        Note: this is the simplest correct policy. Production-quality MSCKF
        variants (S-MSCKF, OpenVINS) prune by keyframe redundancy / parallax
        instead of pure age, which keeps more informative geometry in the window.
        """
        while len(self.cam_states) > max_window_size:
            self._drop_cam_state(0)

    def _drop_cam_state(self, idx):
        """Remove the cam state at list index `idx` and its 6x6 block from P."""
        base = 15 + 6 * idx
        n    = self.P_matrix.shape[0]
        keep = np.r_[0:base, base + 6:n]

        self.P_matrix = self.P_matrix[np.ix_(keep, keep)]
        self.cam_states.pop(idx)

    # def update(self, retired_tracks):
    #     """No gates. DLT triangulation + Mourikis null-space projection only."""
    #     if not retired_tracks:
    #         return
    #     if self.K is None:
    #         raise RuntimeError("MSCKF.update() requires K.")

    #     sigma_norm = self.pixel_noise_std / self.K[0, 0]
    #     H_blocks, r_blocks = [], []

    #     for track in retired_tracks:
    #         valid = [(self._cam_state_index(fid), kp)
    #                 for fid, kp in zip(track.frame_ids, track.keypoints)]
    #         valid = [(idx, kp) for idx, kp in valid if idx is not None]
    #         if len(valid) < 2:
    #             continue

    #         cs_list  = [self.cam_states[idx] for idx, _ in valid]
    #         pixels   = np.array([kp for _, kp in valid], dtype=np.float64)
    #         norm_obs = self.pixels_to_normalized(pixels)

    #         # DLT (no parallax check, no GN refinement)
    #         p_f = triangulate_dlt(norm_obs, cs_list)
    #         if p_f is None:
    #             continue

    #         # Per-view Jacobian — disable depth bound check inside
    #         jacs = self._compute_track_jacobians(track, p_f)
    #         if jacs is None:
    #             continue
    #         H_x, H_f, r = jacs

    #         # Null-space projection (still needed mathematically)
    #         H_o, r_o = self._left_nullspace_project(H_x, H_f, r)
    #         if H_o.shape[0] < 1:
    #             continue

    #         # No chi² gate
    #         H_blocks.append(H_o)
    #         r_blocks.append(r_o)

    #     print(f"[naive update] tracks_in={len(retired_tracks)} used={len(H_blocks)}")

    #     if not H_blocks:
    #         return

    #     H = np.vstack(H_blocks)
    #     r = np.concatenate(r_blocks)

    #     if H.shape[0] > H.shape[1]:
    #         Q, T = np.linalg.qr(H, mode='reduced')
    #         H = T
    #         r = Q.T @ r

    #     R_meas = (sigma_norm ** 2) * np.eye(H.shape[0])
    #     P = self.P_matrix
    #     S = H @ P @ H.T + R_meas
    #     K_gain = P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
    #     delta_x = K_gain @ r

    #     # --- BIAS LOCK ---
    #     # Camera update'leri bias'ı düzeltmesin. Bias static init'te kestirildi,
    #     # kamera observations'ı attitude/vel/pos'a kanalize et — bias absorbing
    #     # observability lock-in'i kırmak için.
    #     # delta_x[3:6]   = 0.0   # δb_g
    #     # delta_x[9:12]  = 0.0   # δb_a


    #     self.nominal_rot  = self.nominal_rot * R.from_rotvec(delta_x[0:3])
    #     self.bg          += delta_x[3:6]
    #     self.nominal_vel += delta_x[6:9]
    #     self.ba          += delta_x[9:12]
    #     self.nominal_pos += delta_x[12:15]

    #     for i, cs in enumerate(self.cam_states):
    #         base = 15 + 6 * i
    #         cs.rot = cs.rot * R.from_rotvec(delta_x[base:base+3])
    #         cs.pos = cs.pos + delta_x[base+3:base+6]

    #     n_state = P.shape[0]
    #     I_KH = np.eye(n_state) - K_gain @ H
    #     P_new = I_KH @ P @ I_KH.T + K_gain @ R_meas @ K_gain.T
    #     self.P_matrix = 0.5 * (P_new + P_new.T)


    # =================================================================
    # LEGACY: batched EKF update (kept for reference)
    # =================================================================
    # Önceden kullandığımız "tek seferde tüm track'leri stack'leyip uygula"
    # versiyonu. Termal'de KLT ile büyük dead_tracks dump'larında (örn.
    # marginalize_at_prune sonrası 100-200 track aynı anda) P_matrix tek
    # adımda dramatik küçülüyordu → sonraki update'lerde chi2 cascade →
    # filter ölçümleri reject etmeye başlıyor → dead-reckoning + drift.
    #
    # Yerine SEQUENTIAL update'e geçtik (aşağıda). Bu blok referans/karşılaştırma
    # için yorum olarak duruyor.
    # -----------------------------------------------------------------
    # def update(self, retired_tracks):
    #     if not retired_tracks:
    #         return
    #     if self.K is None:
    #         raise RuntimeError("MSCKF.update() requires camera intrinsics K.")
    #
    #     sigma_norm = self.pixel_noise_std / self.K[0, 0]
    #     H_blocks, r_blocks = [], []
    #
    #     for track in retired_tracks:
    #         valid = [(self._cam_state_index(fid), kp)
    #                  for fid, kp in zip(track.frame_ids, track.keypoints)]
    #         valid = [(idx, kp) for idx, kp in valid if idx is not None]
    #         if len(valid) < 2:
    #             continue
    #         cs_list  = [self.cam_states[idx] for idx, _ in valid]
    #         pixels   = np.array([kp for _, kp in valid], dtype=np.float64)
    #         norm_obs = self.pixels_to_normalized(pixels)
    #         if self._max_parallax(cs_list, norm_obs) < self.min_parallax_rad:
    #             continue
    #         p_f = triangulate_dlt(norm_obs, cs_list)
    #         if p_f is None: continue
    #         p_f = self._gauss_newton_refine(p_f, cs_list, norm_obs)
    #         if p_f is None: continue
    #         jacs = self._compute_track_jacobians(track, p_f)
    #         if jacs is None: continue
    #         H_x, H_f, r = jacs
    #         H_o, r_o = self._left_nullspace_project(H_x, H_f, r)
    #         df = H_o.shape[0]
    #         if df < 1: continue
    #         thresh = self._chi2_thresh.get(df) or float(_chi2_dist.ppf(0.95, df))
    #         if self._track_mahalanobis_sq(H_o, r_o, sigma_norm) > thresh:
    #             continue
    #         H_blocks.append(H_o)
    #         r_blocks.append(r_o)
    #
    #     if not H_blocks: return
    #
    #     H = np.vstack(H_blocks)
    #     r = np.concatenate(r_blocks)
    #     # Mourikis §III-D QR reduction
    #     if H.shape[0] > H.shape[1]:
    #         Q, T = np.linalg.qr(H, mode='reduced')
    #         H = T
    #         r = Q.T @ r
    #     R_meas = (sigma_norm ** 2) * np.eye(H.shape[0])
    #     P  = self.P_matrix
    #     S  = H @ P @ H.T + R_meas
    #     K_gain = P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
    #     delta_x = K_gain @ r
    #     # ... boxplus + Joseph (tek seferde)
    # =================================================================


    def update(self, retired_tracks):
        """
        SEQUENTIAL EKF measurement update.

        Mourikis 2007'nin gating pipeline'ı (1-7) aynı, AMA accepted
        (H_o, r_o) blokları tek bir dev (H, r)'de stack'lenip uygulanmıyor —
        her track tek tek, kendi küçük Kalman gain'i ile P_matrix'i adım
        adım güncelliyor.

        ----------------------------------------------------------------
        NİYE SEQUENTIAL?
        ----------------------------------------------------------------
        Batched update'te bir frame'de 100-200 track marjinalize olursa:
            stacked H : (Σ df, n_state)
            S         : (Σ df, Σ df)
            K_gain    : (n_state, Σ df)
            ΔP        : K · H · P  → DEV bir tek-adım korreksiyonu
            P_new     : (I - KH) P (I - KH)^T + K R K^T

        Σ df ~ 1000-5000'e ulaşınca, P_new diagonal'ı 10⁴ kat düşer.
        Bir sonraki update'in chi2 gate'i S = H P H^T + R'yi hesaplarken
        P çok küçük olduğu için S tiny, mahalanobis^2 = r^T S^-1 r tavan
        yapar, normal innovation bile dış olarak reddedilir → "chi2 cascade"
        → filter tüm ölçümleri reject eder → IMU dead-reckoning'e döner →
        bias drift'i kontrol edilemez → trajektörü patlatır.

        Sequential update'te:
            * Her track ayrı küçük K_gain getirir (df ~ 1-37).
            * P kademeli küçülür — bir adımda 10² kat değil, yüz küçük adımda
              kontrollü şekilde.
            * Joseph form her step'te symmetry + positive-definiteness korur.
            * Bir kötü track diğerlerini etkileyemez (kendi adımıyla sınırlı).

        ----------------------------------------------------------------
        MATEMATIK — PER-TRACK ADIM
        ----------------------------------------------------------------
        Bir track'in null-space projection sonrası ölçüm modeli:
            r_o = H_o · δx + n_o,    cov(n_o) = R = σ² · I_df

        EKF update bu track için:
            S       = H_o · P · H_o^T + R              # (df × df) innovation cov
            K       = P · H_o^T · S⁻¹                   # (n_state × df) Kalman gain
            δx̂      = K · r_o                           # (n_state,) error-state correction
            P_new   = (I - K H_o) P (I - K H_o)^T       # Joseph form, positive-def stable
                    + K R K^T

        Boxplus (⊞) ile nominal state'e uygula:
            θ_imu  ← θ_imu ⊞ δθ̂_imu      (right-multiplicative quaternion update)
            b_g    ← b_g + δb_g           (additive)
            v      ← v   + δv             (additive)
            b_a    ← b_a + δb_a           (additive)
            p      ← p   + δp             (additive)
            (her cam state için aynı şekilde)

        ----------------------------------------------------------------
        PARAMETRELER (constructor'dan geliyor)
        ----------------------------------------------------------------
        pixel_noise_std    : px, normalized coords için σ = pixel_noise_std/fx
                             Termal KLT için 3-6 makul (per-view drift'i de
                             modellemek istediğimizden gerçek subpixel'in üstünde)
        min_parallax_rad   : DLT ve null-space numerik kararlılığı için minimum
                             bearing-ray spread. < 0.3° → triangulation çöker.
        min_depth/max_depth: GN refinement bounds. min ~10cm (yakın engelleri
                             dışla), max ~500m (sky/uzak landmark filtresi).
        gn_max_iter        : İnverse-depth GN iterasyon limiti. 4-5 yeterli;
                             küçük step'lerle exit early.
        chi2_alpha         : Per-track Mahalanobis gate confidence. 0.99 →
                             %1 yanlış-reject oranı. Daha gevşek (0.999) chi2
                             cascade'i geciktirir ama outlier'lara açık olur.
        """
        if not retired_tracks:
            return
        if self.K is None:
            raise RuntimeError("MSCKF.update() requires camera intrinsics K.")

        # Normalized image-plane'de pixel noise std. Tüm Jacobian'lar K^-1 ile
        # bearing'e geçtikten sonra hesaplandığı için R_meas de aynı çerçevede.
        # fx ≈ K[0,0]; SThereo termal için ~414, FIReStereo için ~406.
        sigma_norm = self.pixel_noise_std / self.K[0, 0]
        sigma_norm_sq = sigma_norm ** 2




        # ── TANI ENSTRÜMANTASYONU — HATA TESPİTİ AMAÇLI EKLENDİ ──────────────
        # Geçici tanı kodu. İlk 5 update çağrısında (a) ham reprojection
        # residual'ini |z - ẑ| ve (b) uygulanan Δx düzeltmesini yazdırır.
        # Amaç: kamera update'inin state'i ilk anlardan itibaren neden bozduğunu
        # lokalize etmek —
        #   büyük |r|              → geometri/extrinsic hatası (üçgenlenen
        #                            landmark ölçüm pikseline düşmüyor)
        #   küçük |r| ama büyük Δx → EKF update matematiği hatası (gain /
        #                            null-space / kovaryans)
        # Sorun bulununca: bu blok + aşağıdaki "_diag" kancaları silinmeli.
        # Pencere: ilk 5 *track kabul eden* update. Boş (used=0) update'ler
        # sayacı yakmasın — yoksa araç sabitken parallax gate'in hepsini elediği
        # ilk kareler pencereyi tüketir, asıl bakmak istediğimiz update'ler
        # diag'sız kalır. Sayaç update() sonunda, accepted doluysa artıyor.
        self._diag_done = getattr(self, '_diag_done', 0)
        _diag_on  = self._diag_done < 5
        _diag_res = []   # accepted track'lerin ham residual vektörleri
        _diag_geo = []   # accepted track başına (M, derinlik_m, parallax_deg)




        # =============================================================
        # PHASE 1 — Gating + linearization (state DEĞİŞMİYOR)
        # =============================================================
        # Tüm track'leri 7-aşamalı pipeline'dan geçir. Geçenlerin (H_o, r_o)
        # bloklarını biriktir. State (nominal pose, bias, P) BURADA değişmiyor
        # — linearization tek bir noktada (update başlangıcındaki state'te)
        # yapılır, sonra Phase 2'de sequential apply edilir.
        accepted = []   # list of (H_o, r_o) tuples; r_o normalized residuals

        n_in        = len(retired_tracks)
        n_short     = 0
        n_parallax  = 0
        n_dlt       = 0
        n_gn        = 0
        n_jac       = 0
        n_chi2      = 0

        for track in retired_tracks:
            # 1. Sliding window içindeki gözlemleri seç (eski cam state silinmiş
            #    olabilir, bunlar artık MSCKF'e bağlanamaz).
            valid = [(self._cam_state_index(fid), kp)
                     for fid, kp in zip(track.frame_ids, track.keypoints)]
            valid = [(idx, kp) for idx, kp in valid if idx is not None]
            if len(valid) < 2:
                n_short += 1
                continue

            cs_list  = [self.cam_states[idx] for idx, _ in valid]
            pixels   = np.array([kp for _, kp in valid], dtype=np.float64)
            norm_obs = self.pixels_to_normalized(pixels)

            # 2. Parallax gate — bearing-ray spread çok düşükse DLT singular,
            #    null-space projection sahte yönler üretir.
            if self._max_parallax(cs_list, norm_obs) < self.min_parallax_rad:
                n_parallax += 1
                continue

            # 3. DLT triangulation — linear, hızlı initial guess.
            p_f = triangulate_dlt(norm_obs, cs_list)
            if p_f is None:
                n_dlt += 1
                continue

            # 4. Inverse-depth GN — reprojection error minimize eden refine;
            #    aynı zamanda min/max depth bound'larını içeride uygular.
            p_f = self._gauss_newton_refine(p_f, cs_list, norm_obs)
            if p_f is None:
                n_gn += 1
                continue

            # 5. Per-view Jacobian: H_x (∂h/∂x_err), H_f (∂h/∂p_f), r = z - h.
            #    FEJ aktif — Jacobian linearization cs.rot_fej/pos_fej'de.
            jacs = self._compute_track_jacobians(track, p_f)
            if jacs is None:
                n_jac += 1
                continue
            H_x, H_f, r = jacs

            # 6. Null-space projection: H_f'i ele (landmark'ı state'ten çıkar).
            #    Sonuç: r_o = H_o · δx + n_o, df = 2M - 3 satır.
            H_o, r_o = self._left_nullspace_project(H_x, H_f, r)
            df = H_o.shape[0]
            if df < 1:
                continue

            # 7. Chi-square gate. UYARI: bu CURRENT P üzerinden bakılıyor —
            #    Phase 2'de henüz hiçbir update uygulanmadı. Sequential'da
            #    ilerleyen track'ler için P küçülecek, ama gate'i tek başına
            #    Phase 1'de değerlendirmek deterministic ve düzenli.
            # TANI (Adım 3): chi2_enabled=False iken gate atlanır — chi2
            # reddinin divergence mekanizmasının parçası olup olmadığını
            # izole etmek için. Kalıcı değil.
            thresh = self._chi2_thresh.get(df) or float(_chi2_dist.ppf(0.95, df))
            if getattr(self, 'chi2_enabled', True) and \
                    self._track_mahalanobis_sq(H_o, r_o, sigma_norm) > thresh:
                n_chi2 += 1
                continue

            accepted.append((H_o, r_o))





            if _diag_on:                       # TANI: ham residual + geometri sakla
                _diag_res.append(r)
                # K patlamasının kök nedenini ayırmak için track geometrisi:
                # derinlik (uzak feature → küçük H → büyük K) ve parallax
                # (düşük parallax → belirsiz üçgenleme). cs_list[0] çapa kamera.
                cs0   = cs_list[0]
                depth = float((cs0.rot.as_matrix().T @ (p_f - cs0.pos))[2])
                prlx  = np.degrees(self._max_parallax(cs_list, norm_obs))
                _diag_geo.append((len(valid), depth, prlx))





        n_used = len(accepted)
        print(f"[update] in={n_in:3d} used={n_used:3d} used/in: {n_used / n_in :.2f}  drops: short={n_short} "
              f"parallax={n_parallax} dlt={n_dlt} gn={n_gn} jac={n_jac} chi2={n_chi2}",
              flush=True)





        # TANI: accepted track'lerin ham reprojection residual'i. İlk karelerde
        # IMU pozları neredeyse kusursuz olduğu için bu küçük olmalı (~birkaç px).
        if _diag_on and _diag_res:
            allr = np.abs(np.concatenate(_diag_res))
            fx   = self.K[0, 0]
            print(f"[diag u{self._diag_done + 1}] ham residual |z-zhat| (normalize): "
                  f"mean={allr.mean():.4f} max={allr.max():.4f}  "
                  f"≈ mean={allr.mean()*fx:.1f}px max={allr.max()*fx:.1f}px", flush=True)
            # P'nin update ÖNCESİ büyüklüğü — K'nın P'den mi (erken = büyük P)
            # yoksa M'den mi (uzun track) patladığını ayırmak için.
            # sqrt(diag(P)) = o eksendeki 1σ belirsizlik.
            Pd = np.sqrt(np.clip(np.diag(self.P_matrix), 0.0, None))
            print(f"[diag u{self._diag_done + 1}] P update-öncesi (1σ): "
                  f"att={np.linalg.norm(Pd[0:3]):.3f}rad "
                  f"bg={np.linalg.norm(Pd[3:6]):.4f} "
                  f"vel={np.linalg.norm(Pd[6:9]):.3f}m/s "
                  f"ba={np.linalg.norm(Pd[9:12]):.4f} "
                  f"pos={np.linalg.norm(Pd[12:15]):.3f}m", flush=True)





        if not accepted:
            return

        # =============================================================
        # PHASE 2 — Sequential per-track Joseph update
        # =============================================================
        # Her accepted (H_o, r_o) için bağımsız küçük EKF step. Track sırası
        # teorik olarak sonucu etkiler (linearizasyon farkları), ama
        # Mourikis tarzı tek-iteration EKF için bu ihmal edilebilir.
        n_state = self.P_matrix.shape[0]
        I_full  = np.eye(n_state)





        _diag_dx    = np.zeros(15)  # TANI: bu çağrıda IMU-state'e uygulanan Σ|Δx|

        # Biriken hata-durumu (accumulated error state). Doğru sequential
        # update: her track'in residual'i o ana kadar uygulanan toplam
        # düzeltmeyle telafi edilir; nominal state'e döngü SONUNDA tek
        # seferde uygulanır. dx_total bu toplamı tutar.
        dx_total    = np.zeros(n_state)
        _diag_condS = []            # TANI: track başına innovation-kovaryans koşul sayısı
        _diag_normK = []            # TANI: track başına Kalman gain normu





        for _trk_i, (H_o, r_o) in enumerate(accepted):
            df = H_o.shape[0]
            R_meas = sigma_norm_sq * np.eye(df)

            # Innovation covariance:  S = H P H^T + R   (df × df, küçük matris)
            S = H_o @ self.P_matrix @ H_o.T + R_meas

            # Kalman gain:  K = P H^T S^-1
            # linear-solve, çünkü direct inversion daha az kararlı.
            K_gain = self.P_matrix @ H_o.T @ np.linalg.solve(S, np.eye(df))

            # Residual'i biriken düzeltmeye göre telafi et:
            #   r_active = r_o - H_o · dx_total
            # Bu satır olmadan her track önceki track'lerin çözdüğü hatayı
            # yeniden görüp yeniden düzeltir → pozitif geri besleme →
            # overshoot → diverge.
            r_active = r_o - H_o @ dx_total

            # Bu adımın hata-durumu katkısı:  δx_step = K · r_active
            delta_x = K_gain @ r_active

            # ── TANI: ATTITUDE-LOCK ──────────────────────────────────────
            # Açıkken update IMU attitude'unu (δθ) düzeltmez. İzolasyon
            # deneyi: spurious attitude tekmesi runaway'in sebebi mi?
            if getattr(self, 'lock_imu_attitude', False):
                delta_x[0:3] = 0.0

            # Bu adımın katkısını biriken toplama ekle (nominal'e DEĞİL).
            dx_total += delta_x




            if _diag_on:                       # TANI: gain / koşulluluk ölç
                _diag_dx += np.abs(delta_x[:15])
                _diag_condS.append(float(np.linalg.cond(S)))
                _diag_normK.append(float(np.linalg.norm(K_gain)))
                # İlk birkaç track'i tek tek dök: hangi track K'yı patlatıyor,
                # ve o track uzak / düşük-parallax mı?
                if _trk_i < 6:
                    M_obs, depth, prlx = _diag_geo[_trk_i]
                    # 1a/1b ayrımı: kamera-state'lerin kendi attitude düzeltmesi
                    # (δθ_cam) ile IMU attitude düzeltmesini (δθ_imu) karşılaştır.
                    #   δθ_imu >> δθ_cam → kovaryans bağı IMU'yu orantısız
                    #                      şişiriyor (1b — coupling hatası)
                    #   δθ_imu ≈ δθ_cam, ikisi de büyük → ölçüm büyük attitude
                    #                      düzeltmesi dayatıyor (1a — kamera tarafı)
                    n_cam = len(self.cam_states)
                    cam_dth = max((np.linalg.norm(delta_x[15+6*i:15+6*i+3])
                                   for i in range(n_cam)), default=0.0)
                    cam_dp  = max((np.linalg.norm(delta_x[15+6*i+3:15+6*i+6])
                                   for i in range(n_cam)), default=0.0)
                    print(f"[diag u{self._diag_done + 1}]   track#{_trk_i}: "
                          f"M={M_obs} depth={depth:7.1f}m parallax={prlx:5.2f}deg "
                          f"||K||={np.linalg.norm(K_gain):7.1e}  "
                          f"IMU dth={np.linalg.norm(delta_x[0:3]):.4f}rad "
                          f"dp={np.linalg.norm(delta_x[12:15]):.3f}m  "
                          f"CAM max-dth={cam_dth:.4f}rad max-dp={cam_dp:.3f}m", flush=True)




            # ----- JOSEPH-FORM COVARIANCE UPDATE -----
            # P = (I - KH) P (I - KH)^T + K R K^T
            # Naive (I - KH) P formu zamanla asimetrik drift eder ve
            # positive-definiteness kaybedebilir; Joseph form (I - KH)'ın
            # sağdan da çarpılması ve K R K^T eklenmesiyle her step'te PD
            # kalmasını garanti eder.
            I_KH = I_full - K_gain @ H_o
            P_new = I_KH @ self.P_matrix @ I_KH.T + K_gain @ R_meas @ K_gain.T

            # Sayısal asimetri biriken yuvarlama hatalarından gelir; her
            # adımda 0.5 * (P + P^T) ile düzelt.
            self.P_matrix = 0.5 * (P_new + P_new.T)

        # ── TANI: REPROJECTION RESIDUAL AZALMASI ─────────────────────────
        # Update lokal olarak doğru çalışıyor mu? Toplam ||r||_pre vs
        # ||r||_post karşılaştır. dx_total uygulandıktan sonra her accepted
        # track'in yeni residual'i r_post = r_o − H_o · dx_total. Sağlıklı
        # bir EKF update bunu AZALTIR; oran > 1 ise update math'inde ters
        # yön/işaret hatası vardır — bug lokalize.
        if _diag_on:
            sum_pre  = sum(np.linalg.norm(r) for _, r in accepted)
            sum_post = sum(np.linalg.norm(r - H @ dx_total) for H, r in accepted)
            ratio    = sum_post / sum_pre if sum_pre > 0 else float('nan')
            tag      = 'AZALDI' if ratio < 1.0 else 'ARTTI'
            print(f"[diag u{self._diag_done + 1}] residual reduction: "
                  f"||r||_pre={sum_pre:.3f} → ||r||_post={sum_post:.3f}  "
                  f"ratio={ratio:.3f}  ({tag})", flush=True)

        # ----- BOXPLUS: biriken toplam Δx'i nominal state'e BİR KERE uygula.
        # Nominal state döngü İÇİNDE güncellenmez — yoksa H_o/r_o eski
        # linearizasyon noktasına göre bayatlar. Döngü bitti, hepsi tutarlı.
        self.nominal_rot  = self.nominal_rot * R.from_rotvec(dx_total[0:3])
        self.bg          += dx_total[3:6]
        self.nominal_vel += dx_total[6:9]
        self.ba          += dx_total[9:12]
        self.nominal_pos += dx_total[12:15]

        for i, cs in enumerate(self.cam_states):
            base = 15 + 6 * i
            cs.rot = cs.rot * R.from_rotvec(dx_total[base:base + 3])
            cs.pos = cs.pos + dx_total[base + 3:base + 6]





        # TANI: bu update çağrısında IMU-state'e uygulanan toplam düzeltme.
        # δθ büyükse attitude tekmeleniyor → gravity sızıyor → z/hız patlıyor.
        if _diag_on:
            print(f"[diag u{self._diag_done + 1}] uygulanan toplam Δx (IMU state): "
                  f"dtheta={np.linalg.norm(dx_total[0:3]):.4f}rad "
                  f"dbg={np.linalg.norm(dx_total[3:6]):.5f} "
                  f"dv={np.linalg.norm(dx_total[6:9]):.4f}m/s "
                  f"dba={np.linalg.norm(dx_total[9:12]):.5f} "
                  f"dp={np.linalg.norm(dx_total[12:15]):.4f}m", flush=True)
            # cond(S): "sıfıra bölmeye ne kadar yakınız". >~1e6 → kötü-koşullu,
            # belirsiz-derinlik track'i → gain patlıyor. ||K|| doğrudan gain büyüklüğü.
            if _diag_condS:
                print(f"[diag u{self._diag_done + 1}] track-bazli: "
                      f"cond(S) max={max(_diag_condS):.2e} min={min(_diag_condS):.2e}  "
                      f"||K|| max={max(_diag_normK):.2e}", flush=True)

        # Buraya yalnızca accepted doluyken ulaşılır (boşsa yukarıda return
        # edilmişti). Demek ki bu update gerçekten track uyguladı — diag
        # penceresinden bir slot harca.
        if _diag_on:
            self._diag_done += 1





    @staticmethod
    def _left_nullspace_project(H_x, H_f, r):
        """
        Project a per-track measurement system onto the left null-space of H_f
        to eliminate the unknown landmark — the core Mourikis 2007 MSCKF trick.

        Given:
            H_x: (2M, n_state)  — derivative w.r.t. error state
            H_f: (2M, 3)        — derivative w.r.t. landmark
            r  : (2M,)          — residual

        Compute the full QR  H_f = [Q1 Q2] · [R1; 0]  and premultiply everything
        by Q2^T.  Q2 spans the (2M-3)-dim left null-space of H_f, so the
        landmark term vanishes and the measurement equation depends only on δx:

            r_o = H_o · δx + n_o,    H_o = Q2^T · H_x,    r_o = Q2^T · r

        Returns:
            (H_o, r_o) with shapes ((2M-3, n_state), (2M-3,)).

        Caveat: this assumes H_f has full column rank (3). Triangulation that
        is near-degenerate (zero parallax) violates this — a parallax check
        BEFORE calling this is what makes MSCKF consistent in practice.
        """
        # mode='complete' gives the full 2M x 2M Q; mode='reduced' would only
        # return Q1 (the first 3 columns) and we need Q2.
        Q, _ = np.linalg.qr(H_f, mode='complete')
        Q2 = Q[:, 3:]
        H_o = Q2.T @ H_x
        r_o = Q2.T @ r
        return H_o, r_o

    def pixels_to_normalized(self, pixels):
        """Convert raw pixel observations to normalized image-plane coordinates
        (i.e. [x/z, y/z] = K^-1 · [u, v, 1]). If self.D is set, distortion is
        also undone — use this when the dataloader did NOT image-level undistort.

        Args:
            pixels: (N, 2) array of (u, v) pixel coordinates.

        Returns:
            (N, 2) array of normalized observations.
        """
        if self.K is None:
            raise RuntimeError("MSCKF.K is not set — pass K to the constructor.")

        pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        # cv2.undistortPoints with P=None returns normalized coords directly.
        # D=None / zeros → pure K^-1 (no distortion correction).
        norm = cv2.undistortPoints(pts.astype(np.float32), self.K, self.D)
        return norm.reshape(-1, 2).astype(np.float64)
