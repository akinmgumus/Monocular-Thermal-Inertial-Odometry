"""Dataset paths, calibration loaders, and IMU noise constants for ROVTIO.

ROVTIO dataset (NTNU ARL): real thermal-inertial recordings on a DJI Matrice 100
("charlie" drone) with:
  - FLIR Tau2 thermal camera (640x512, 16-bit raw, ~25 Hz)
  - VectorNav VN-100 IMU (~168 Hz)
  - Vicon motion capture ground truth (~6 Hz)
  - Indoor scene (Vicon requires controlled environment)

Source repo: https://github.com/ntnu-arl/rovtio
Dataset:     https://huggingface.co/datasets/ntnu-arl/rovtio

Calibration values below are taken directly from the ROVTIO config files
distributed with the source repo (cfg/rovtio/charlie_tau2.yaml and
cfg/rovtio/rovtio.info). The dataset authors calibrated this rig themselves;
we use their values without modification.
"""

import os
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


# ----------------------------------------------------------------------
# DATASET LAYOUT — ROVTIO alt1
# ----------------------------------------------------------------------
DATA_ROOT     = os.path.expanduser('~/rovtio_data')
PROCESSED_DIR = os.path.join(DATA_ROOT, 'processed', 'alt1')
# Raw bag files location (10 split .bag files for one recording). The
# extractor reads from here; the pipeline reads from PROCESSED_DIR.
BAG_DIR       = os.path.expanduser('~/alt1')

CAM0_DIR      = os.path.join(PROCESSED_DIR, 'cam0', 'data')
CAM0_TS       = os.path.join(PROCESSED_DIR, 'cam0', 'timestamps.txt')
IMU_CSV       = os.path.join(PROCESSED_DIR, 'imu0', 'data.csv')
GT_TRAJ       = os.path.join(PROCESSED_DIR, 'gt', 'traj.txt')


# ----------------------------------------------------------------------
# THERMAL CAMERA INTRINSICS  (FLIR Tau2, from charlie_tau2.yaml)
# ----------------------------------------------------------------------
def load_camera_intrinsics():
    """Tau2 thermal camera intrinsics.

    Distortion model is 'equidistant' (Kalibr / fisheye), NOT plumb_bob/radtan.
    Use cv2.fisheye.undistortImage / initUndistortRectifyMap to undistort.
    """
    K = np.array([[440.381678762, 0.0,           328.082910449],
                  [  0.0,         440.34105427,  257.335140534],
                  [  0.0,         0.0,           1.0]], dtype=np.float64)
    # 4-element equidistant distortion (k1, k2, k3, k4).
    D = np.array([-0.125172452973, -0.0158508045529,
                   0.00552041769855, 0.00188160221577], dtype=np.float64)
    return {
        'K': K,
        'D': D,
        'width': 640,
        'height': 512,
        'distortion_model': 'equidistant',
    }


# ----------------------------------------------------------------------
# IMU → CAMERA EXTRINSIC  (Camera1 block in rovtio.info)
# ----------------------------------------------------------------------
# The ROVTIO config provides:
#   qCM  = IMU → Camera rotation, Hamilton (x, y, z, w)
#   MrMC = IMU → Camera translation vector, expressed in IMU frame [m]
# We need T_imu_cam (the 4x4 transform such that p_imu = T_imu_cam @ p_cam).
#   - translation part   t_imu_cam = MrMC (already in IMU frame)
#   - rotation part      R_imu_cam = R(qCM)^T  (because qCM rotates IMU→Cam,
#                                                 so its inverse maps Cam→IMU)
_Q_CAM_IMU_HAMILTON_XYZW = (-0.575090212567,  0.57200349159,
                             -0.413485129831, -0.413658434494)
_T_IMU_CAM_IN_IMU        = np.array([0.132385017016,
                                      0.0231335126268,
                                     -0.0421824381085])


def load_extrinsic_imu_cam():
    """Return T_imu_cam (4x4)."""
    R_cam_imu = R.from_quat(_Q_CAM_IMU_HAMILTON_XYZW).as_matrix()
    R_imu_cam = R_cam_imu.T
    T_imu_cam = np.eye(4)
    T_imu_cam[:3, :3] = R_imu_cam
    T_imu_cam[:3, 3]  = _T_IMU_CAM_IN_IMU
    return T_imu_cam


def load_timeshift_imu_cam():
    """Camera-IMU time offset in seconds.

    From rovtio.launch: cam1_offset = -0.02414188675155223 with the convention
    `t_imu = t_cam + shift`. So t_imu = t_cam - 0.0241 s — the IMU clock is
    24 ms BEHIND the camera. To align IMU to camera time, ADD 0.0241 s to
    every IMU timestamp (pass +0.0241 as t_offset to IMULoader).
    """
    return -0.02414188675155223


# ----------------------------------------------------------------------
# IMU NOISE MODEL — VectorNav VN-100
# ----------------------------------------------------------------------
# Values reverse-engineered from rovtio.info filter-tuning constants. The
# ROVTIO authors used these to drive their UKF and they correspond to the
# variance of the CONTINUOUS-time noise densities (units: var, not std).
# We convert to MSCKF's standard sigma (continuous-time noise density).
#
#   vel  prediction noise variance 4.0e-5  m^2/s^3   →  sigma_a  = sqrt(4e-5) ≈ 6.3e-3 m/s²/√Hz
#   att  prediction noise variance 7.6e-8  rad^2/s   →  sigma_g  = sqrt(7.6e-8) ≈ 2.8e-4 rad/s/√Hz
#   acb  prediction noise variance 1.0e-8  m^2/s^5   →  sigma_ba = sqrt(1e-8)  = 1.0e-4 m/s³/√Hz
#   gyb  prediction noise variance 3.8e-9  rad^2/s^3 →  sigma_bg = sqrt(3.8e-9) ≈ 6.2e-5 rad/s²/√Hz
#
# These are reasonable values for VN-100 industrial IMU and match the
# datasheet order of magnitude.
IMU_GYRO_NOISE_DENSITY  = 2.8e-04    # rad / s / sqrt(Hz)
IMU_ACCEL_NOISE_DENSITY = 6.3e-03    # m / s^2 / sqrt(Hz)
IMU_GYRO_BIAS_RW        = 6.2e-05    # rad / s^2 / sqrt(Hz)
IMU_ACCEL_BIAS_RW       = 1.0e-04    # m / s^3 / sqrt(Hz)


# ----------------------------------------------------------------------
# GRAVITY
# ----------------------------------------------------------------------
# rovtio.info uses g_z = -10.0854, which matches the observed accel norm in
# the static segment of this specific recording (sensor scale/local effect).
# We default to standard 9.81 and let the bias init absorb any residual.
# If results show systematic Z drift, switch to 10.0854 here.
GRAVITY_MAGNITUDE = 9.81


# ----------------------------------------------------------------------
# GT LOADER  (Vicon motion capture, TransformStamped → TUM)
# ----------------------------------------------------------------------
def load_ground_truth(gt_path=GT_TRAJ):
    """Load Vicon ground truth trajectory.

    Format: '# timestamp x y z q_x q_y q_z q_w'  (TUM-style, written by
    extract_rovtio_bag.py from /vicon/charlie/charlie TransformStamped).

    This is REAL motion capture ground truth (sub-mm accuracy), unlike
    FIReStereo's LiDAR-IO pseudo-GT or VOXL's PX4 EKF2 self-estimate.

    Returns: dict with t [s], p [Nx3], q [Nx4 in qx qy qz qw order].
    """
    raw = np.loadtxt(gt_path, comments='#')
    return {
        't': raw[:, 0].astype(np.float64),
        'p': raw[:, 1:4].astype(np.float64),
        'q': raw[:, 4:8].astype(np.float64),
    }


# ----------------------------------------------------------------------
# Sampling rates (for sanity checks).
# ----------------------------------------------------------------------
IMU_RATE_HZ       = 168.0
THERMAL_RATE_HZ   =  25.0
VICON_RATE_HZ     =   6.0


# ----------------------------------------------------------------------
# SEGMENT ANALYSIS — fixed thresholds (no auto-tuning); scene-scale dependent
# ----------------------------------------------------------------------
SEG_THRESHOLD_M    = 2.0    # tracking/diverged boundary on 4-DOF-aligned |err|
SEG_MIN_DURATION_S = 2.0     # shorter excursions/dips are transients
