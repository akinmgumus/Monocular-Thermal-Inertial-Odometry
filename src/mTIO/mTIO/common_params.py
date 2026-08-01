"""Shared experiment parameters for all datasets.

Every test script imports these so that the front-end, back-end and preprocessing
configuration is IDENTICAL across datasets. Only two things are chosen per run in
the test script: the front-end (KLT vs ORB) and CLAHE on/off.

Dataset-specific PHYSICAL quantities stay in the per-dataset ``config_*.py`` files
and are NOT tunable parameters, so they are not here:
    - camera intrinsics K, distortion D and the distortion model,
    - IMU noise (the process-noise Q matrix),
    - IMU-camera time offset, paths, ground-truth loaders, init window.

Usage in a test script::

    from mTIO import common_params as P

    METHOD    = 'orb'      # 'orb' or 'klt'   <- chosen per run
    USE_CLAHE = True        #                  <- chosen per run

    tracker = P.make_tracker(METHOD)
    msckf   = MSCKF(K=K, D=None, **P.MSCKF_PARAMS)
    msckf.Q_matrix = ...                        # per-dataset, from config
    loader  = ThermalDataLoader(cam_dir, bit_depth=16, undistort=True,
                                K=K, D=D, distortion_model=dist_model,
                                use_clahe=USE_CLAHE, **P.PREPROC)

To retune the pipeline, edit THIS file once; every dataset picks it up.
"""

from mTIO.klt import KLTTracker
from mTIO.orb import ORBTracker


# Names exported by ``from mTIO.common_params import *``. Test scripts pull
# the whole shared block through this; the tracker classes above are deliberately
# NOT listed, so they do not leak into the importing namespace.
__all__ = [
    'LOCK_ATTITUDE', 'CHI2_GATE',
    'MAX_WINDOW', 'MAX_TRACKS_PER_UPDATE',
    'ZUPT', 'ZUPT_GYRO_THRESH', 'ZUPT_ACCEL_DEV_THRESH',
    'ZUPT_IMU_WINDOW', 'ZUPT_THROTTLE', 'ZUPT_SIGMA',
    'PREPROC', 'MSCKF_PARAMS', 'KLT_PARAMS', 'ORB_PARAMS',
    'make_tracker',
]


# ── Diagnostic switches ─────────────────────────────────────────────────────
# Fixed for every run; exposed so test scripts pick them up via ``import *``.
LOCK_ATTITUDE = False   # freeze IMU attitude in propagation (debug only)
CHI2_GATE     = True    # Mahalanobis chi-square outlier gate on the update


# ── Sliding window / update ────────────────────────────────────────────────
MAX_WINDOW            = 20     # camera poses kept in the sliding window
MAX_TRACKS_PER_UPDATE = 50     # longest tracks used per measurement update


# ── ZUPT ────────────────────────────────────────────────────────────────────
# Disabled for all core runs. ZUPT is only meaningful on platforms with long
# static/hover phases, and its rest detector can corrupt slow genuine motion;
# it is therefore not part of the standard configuration. The thresholds below
# are the shared defaults used *if* a script flips ZUPT on (norm-based rest
# detector: mean gyro norm < GYRO_THRESH and | |a| - g | < ACCEL_DEV_THRESH).
ZUPT                  = False
ZUPT_GYRO_THRESH      = 0.05   # rad/s   — mean gyro-norm rest threshold
ZUPT_ACCEL_DEV_THRESH = 0.3    # m/s^2   — |accel-norm - g| rest threshold
ZUPT_IMU_WINDOW       = 50     # samples — rest-detection buffer length
ZUPT_THROTTLE         = 10     # call ZUPT every Nth IMU sample when at rest
ZUPT_SIGMA            = 0.01   # m/s     — pseudo zero-velocity measurement std

# Divergence is NOT auto-detected. Failure points (settle / divergence) are
# marked by hand per run in results/divergence_marks.csv and applied at
# evaluation time (see mTIO.evaluation.load_divergence_mark), because the
# transient behaviours differ too much between datasets for a single rule.


# ── Thermal preprocessing ───────────────────────────────────────────────────
# CLAHE on/off is chosen per run (USE_CLAHE); these are the shared operator
# settings. Only affects 16-bit datasets; a no-op for 8-bit (EuRoC).
PREPROC = dict(
    normalize      = 'percentile',
    percentile     = (2.0, 98.0),
    clahe_clip     = 3.0,
    clahe_grid     = (8, 8),
    gaussian_sigma = 1.0,
    gaussian_ksize = (3, 3),
)


# ── MSCKF back-end (shared across datasets) ─────────────────────────────────
# K, D and the process-noise Q are set per dataset (calibration + IMU noise are
# sensor facts) and are therefore NOT included here. Depth bounds use a single
# generous range that is plausible for every scene (indoor near .. outdoor far).
MSCKF_PARAMS = dict(
    pixel_noise_std  = 2.5,
    min_parallax_deg = 0.1,
    min_depth        = 0.1,
    max_depth        = 200.0,
    gn_max_iter      = 4,
    chi2_alpha       = 0.99,
    init_att_std     = 0.01,
    init_bg_std      = 0.005,
    init_vel_std     = 0.05,
    init_ba_std      = 0.05,
    init_pos_std     = 0.01,
)


# ── Front-end parameters (shared across datasets) ───────────────────────────
KLT_PARAMS = dict(
    n_features             = 700,
    fb_eps                 = 1.5,
    ransac_thresh          = 2.0,
    min_track_length       = 3,
    lk_win_size            = (30, 30),
    lk_max_level           = 3,
    quality_level          = 0.01,
    min_distance           = 8.0,
    max_pixel_displacement = 150,
)

ORB_PARAMS = dict(
    n_features             = 1500,   # increased
    grid_rows              = 4,      # decreased
    grid_cols              = 4,      # decreased
    ratio_thresh           = 0.85,   # relaxed
    ransac_thresh          = 3.5,   # increased
    min_track_length       = 4,      # increased
    fast_threshold         = 7,     # increased
    edge_threshold         = 19,    # increased
    max_pixel_displacement = 150,   # decreased
)


def make_tracker(method):
    """Return a KLT or ORB tracker configured with the shared parameters."""
    if method == 'klt':
        return KLTTracker(**KLT_PARAMS)
    if method == 'orb':
        return ORBTracker(**ORB_PARAMS)
    raise ValueError(f"Unknown METHOD={method!r}; use 'klt' or 'orb'.")
