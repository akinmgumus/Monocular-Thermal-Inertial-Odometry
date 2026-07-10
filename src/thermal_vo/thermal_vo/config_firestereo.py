"""Dataset paths, calibration loaders, and IMU noise constants for FIReStereo.

FIReStereo: stereo thermal (FLIR Boson+, 640x512, ~16 fps) + IMU (Epson G365PDC1,
200 Hz) + LiDAR. Outdoor wildfire / urban / forest sequences.
Repo: https://github.com/firestereo/FIReStereo

Note on IMU noise priors:
The FIReStereo authors did NOT publish IMU calibration (no Allan variance, no
Kalibr output). Values below were derived in our pipeline as follows:

  sigma_g, sigma_a (white noise):
    Measured on the 47-second static segment at the end of frick_1/run1.bag
    (t in [164, 211] s). Computed as std(samples) / sqrt(rate). This is
    mathematically equivalent to the left-side slope of an Allan variance
    curve. Inflated by 1.5x for safety margin.

  sigma_bg, sigma_ba (bias random walk):
    NOT derivable from < 1 hour of static data (requires the right-side of
    the Allan curve, which lives at tau > 100 s). Derived instead from the
    Epson G365 datasheet bias instability values (1.2 deg/hr gyro, 14 uG
    accel) assuming a correlation time of ~100 s, and used DIRECTLY without
    any inflation factor: the dataset-provided parametrisation is evaluated
    as-is (RQ3b), since inflating it would mask the drift it is meant to reveal.

  Sanity check: G365 is a higher-grade IMU than ADIS16448 (used in EuRoC),
  so all four values are expected to be lower than EuRoC's, which they are.

  These values should be refined with NIS chi-squared introspection
  (see msckf diagnostic suite) rather than treated as ground truth.
"""

import os
import numpy as np
import yaml


# ----------------------------------------------------------------------
# DATASET LAYOUT — FIReStereo frick_1
# ----------------------------------------------------------------------
DATA_ROOT = os.path.expanduser('~/FIReStereo_data')

# ROS2 converted bag (created from rosbags-convert on the original ROS1 bag)
BAG_PATH  = os.path.join(DATA_ROOT, 'rosbags_ros2', 'frick_1')

# Trajectory ground truth (LiDAR-IO derived, treat as pseudo-GT, not absolute)
GT_TRAJ   = os.path.join(DATA_ROOT, 'reconstruction', 'frick_1', 'traj.txt')

# Calibration file (from cloned repo)
CALIB_YAML = os.path.expanduser('~/FIReStereo/config/firestereo.yaml')


# ROS topic names inside the bag
TOPIC_THERMAL_LEFT  = '/thermal_left/image'
TOPIC_THERMAL_RIGHT = '/thermal_right/image'
TOPIC_IMU           = '/imu/data'


# ----------------------------------------------------------------------
# CALIBRATION LOADERS
# ----------------------------------------------------------------------
def load_camera_intrinsics(cam='cam0', yaml_path=CALIB_YAML):
    """Load thermal_left ('cam0') or thermal_right ('cam1') intrinsics.

    FIReStereo distortion is radtan with 4 coefficients [k1, k2, p1, p2].
    OpenCV expects 5 elements, so k3=0 is appended.

    Both cameras share identical intrinsics in firestereo.yaml.

    Returns: dict(K, D, width, height).
    """
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    c = d[cam]
    fu, fv, cu, cv = c['intrinsics']
    K = np.array([[fu, 0.0, cu],
                  [0.0, fv,  cv],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(c['distortion_coeffs'], dtype=np.float64)
    D = np.array([dist[0], dist[1], dist[2], dist[3], 0.0], dtype=np.float64)
    w, h = c['resolution']
    return {'K': K, 'D': D, 'width': int(w), 'height': int(h)}


def load_extrinsic_imu_cam0(yaml_path=CALIB_YAML):
    """Return T_imu_cam0 (4x4) from firestereo.yaml.

    Convention is the same as EuRoC: p_imu = T_imu_cam0 * p_cam0.
    """
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    return np.array(d['cam0']['T_imu_cam'], dtype=np.float64).reshape(4, 4)


def load_extrinsic_imu_cam1(yaml_path=CALIB_YAML):
    """Return T_imu_cam1 (4x4), derived from T_imu_cam0 and the stereo baseline.

    The yaml does not provide T_imu_cam1 explicitly. We assume the standard
    rectified-stereo convention: cam1 is translated along cam0's +x axis by
    'baseline' meters, with no rotation relative to cam0. This is consistent
    with both cameras having identical intrinsics in the yaml.
    """
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    T_imu_cam0 = np.array(d['cam0']['T_imu_cam'], dtype=np.float64).reshape(4, 4)
    baseline = float(d['baseline'])
    T_cam0_cam1 = np.eye(4)
    T_cam0_cam1[0, 3] = baseline
    return T_imu_cam0 @ T_cam0_cam1


def load_timeshift_imu_cam(yaml_path=CALIB_YAML):
    """Return the IMU-camera time offset in seconds (from firestereo.yaml).

    Definition: t_imu = t_cam + timeshift. FIReStereo value is -0.0386 s.
    This is a large offset (EuRoC is < 5 ms), so MSCKF should treat 'td' as
    an online-estimated state initialized at this value, NOT held fixed.
    """
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    return float(d['cam0']['timeshift_imu_cam'])


def load_baseline(yaml_path=CALIB_YAML):
    """Stereo baseline in meters (~0.2458 m)."""
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    return float(d['baseline'])


def load_ground_truth(gt_path=GT_TRAJ):
    """Load LiDAR-IO pseudo-ground-truth trajectory from reconstruction folder.

    Format (TUM-like): '#timestamp x y z q_x q_y q_z q_w'
    Note: this is NOT absolute ground truth. It is a LiDAR-inertial odometry
    output with its own drift (typically 0.5-2 percent over distance traveled).
    Always report comparisons as 'vs LiDAR-IO reference', not 'vs ground truth'.

    Returns: dict with t [s], p [Nx3], q [Nx4 in qx qy qz qw order].
    """
    raw = np.loadtxt(gt_path, comments='#')
    return {
        't': raw[:, 0].astype(np.float64),               # already in seconds
        'p': raw[:, 1:4].astype(np.float64),
        'q': raw[:, 4:8].astype(np.float64),             # qx qy qz qw
    }


# ----------------------------------------------------------------------
# IMU NOISE MODEL — Epson G365PDC1 (see provenance note at top of file)
# ----------------------------------------------------------------------
# White noise: measured on frick_1 static segment, used DIRECTLY (no safety
# margin). The thesis evaluates the dataset-provided IMU parametrisation as-is
# (RQ3b); inflating it would mask the very effect under study.
IMU_GYRO_NOISE_DENSITY  = 1.06e-04   # rad / s / sqrt(Hz)   (measured, raw)
IMU_ACCEL_NOISE_DENSITY = 8.56e-04   # m / s^2 / sqrt(Hz)   (measured, raw)

# Bias random walk: datasheet bias-instability (tau_corr=100s), used directly.
# Lower than EuRoC's ADIS16448 values, consistent with G365 being a better IMU.
IMU_GYRO_BIAS_RW        = 1.0e-06    # rad / s^2 / sqrt(Hz)  (raw; was x5 inflated)
IMU_ACCEL_BIAS_RW       = 1.0e-04    # m / s^3 / sqrt(Hz)    (raw; was x5 inflated)

# Initial bias estimates from the same static segment used for noise derivation.
# Use these as initial values for the MSCKF bias states b_g and b_a.
IMU_INIT_BIAS_GYRO  = np.array([ 6.55e-04,  2.15e-03, -3.77e-03])  # rad/s
IMU_INIT_BIAS_ACCEL = np.array([-0.9154,    0.1095,    0.0261   ])  # m/s^2

GRAVITY_MAGNITUDE = 9.81

# Sampling rates (for sanity checks, not used directly by MSCKF).
IMU_RATE_HZ        = 200.0
THERMAL_RATE_HZ    =  16.0   # nominal; actual recording is irregular ~15-30 fps

# ----------------------------------------------------------------------
# SEGMENT ANALYSIS — fixed thresholds (no auto-tuning); scene-scale dependent
# ----------------------------------------------------------------------
SEG_THRESHOLD_M    = 2.0    # tracking/diverged boundary on 4-DOF-aligned |err|
SEG_MIN_DURATION_S = 2.0     # shorter excursions/dips are transients
