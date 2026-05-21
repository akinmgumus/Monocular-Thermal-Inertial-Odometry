"""Dataset paths, calibration loaders, and tuning constants for the thermal-VIO pipeline."""

import os
import re
import numpy as np
import cv2


# ----------------------------------------------------------------------
# DATASET LAYOUT — SThereo "valley_evening"
# ----------------------------------------------------------------------
DATA_ROOT = os.path.join(os.path.dirname(__file__), '../../../data/valley_evening')
DATA_ROOT = os.path.abspath(DATA_ROOT)

THERMAL_LEFT_DIR  = os.path.join(DATA_ROOT, 'image/stereo_thermal_14_left')
IMU_CSV           = os.path.join(DATA_ROOT, 'sensor_data/xsens_imu.csv')
CAM_INTRINSIC_YAML = os.path.join(DATA_ROOT, 'calibration/thermal_14bit_left.yaml')
CAM_IMU_EXTRINSIC  = os.path.join(DATA_ROOT, 'calibration/extrinsic/imu_2_thermal_left')
GT_LOCAL_POSE      = os.path.join(DATA_ROOT, 'pose/local_pose.csv')


# ----------------------------------------------------------------------
# CALIBRATION LOADERS
# ----------------------------------------------------------------------
def load_camera_intrinsics(yaml_path=CAM_INTRINSIC_YAML):
    """
    Parse an OpenCV-format YAML camera-calibration file.
    Returns: dict with K (3x3), D (5,), width, height, plus rectification/projection.
    """
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration YAML: {yaml_path}")

    cal = {
        'width':  int(fs.getNode('image_width').real()),
        'height': int(fs.getNode('image_height').real()),
        'K':      fs.getNode('camera_matrix').mat(),
        'D':      fs.getNode('distortion_coefficients').mat().ravel(),
        'R':      fs.getNode('rectification_matrix').mat(),
        'P':      fs.getNode('projection_matrix').mat(),
    }
    fs.release()
    return cal


def load_extrinsic(path=CAM_IMU_EXTRINSIC):
    """
    Parse SThereo's plain-text 4x4 extrinsic file (numbers wrapped in brackets).
    Returns the 4x4 homogeneous transform as a numpy array.

    The file imu_2_thermal_left contains T_thermal_imu, i.e. it maps a point
    from the IMU frame into the thermal-camera frame:
        p_thermal = R · p_imu + t
    """
    with open(path, 'r') as f:
        text = f.read()
    nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', text)]
    if len(nums) != 16:
        raise ValueError(f"Expected 16 values in {path}, got {len(nums)}")
    return np.array(nums, dtype=np.float64).reshape(4, 4)


# ----------------------------------------------------------------------
# IMU NOISE MODEL — Xsens MTi-300 datasheet, used in MSCKF.Q
# ----------------------------------------------------------------------
IMU_GYRO_NOISE_DENSITY  = 0.0001745   # rad/s/√Hz
IMU_ACCEL_NOISE_DENSITY = 0.0005886   # m/s²/√Hz
IMU_GYRO_BIAS_RW        = 0.0000048   # rad/s/√s
IMU_ACCEL_BIAS_RW       = 0.0001471   # m/s²/√s 


# ----------------------------------------------------------------------
# PIPELINE TUNING
# ----------------------------------------------------------------------
STATIC_INIT_SECONDS = 3.5       # seconds of stationary IMU used to bootstrap state
                                # IMU analizi (gyro_max & accel_std) datasette
                                # motion onset'in t≈6.0s'de keskin başladığını
                                # gösteriyor; 5.0s güvenli durağan pencere.
GRAVITY_MAGNITUDE   = 9.81      # m/s² (overridden by initialize_from_static)
