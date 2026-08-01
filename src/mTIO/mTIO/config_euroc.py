"""Dataset paths, calibration loaders, and IMU noise constants for EuRoC MAV."""

import os
import numpy as np
import yaml


# ----------------------------------------------------------------------
# DATASET LAYOUT — EuRoC MH_03_medium
# ----------------------------------------------------------------------
DATA_ROOT = os.path.expanduser('~/machine_hall/MH_03_medium/MH_03_medium')
MAV0      = os.path.join(DATA_ROOT, 'mav0')

CAM0_DIR  = os.path.join(MAV0, 'cam0', 'data')
CAM0_YAML = os.path.join(MAV0, 'cam0', 'sensor.yaml')
IMU_CSV   = os.path.join(MAV0, 'imu0', 'data.csv')
IMU_YAML  = os.path.join(MAV0, 'imu0', 'sensor.yaml')
GT_CSV    = os.path.join(MAV0, 'state_groundtruth_estimate0', 'data.csv')


# ----------------------------------------------------------------------
# CALIBRATION LOADERS
# ----------------------------------------------------------------------
def load_camera_intrinsics(yaml_path=CAM0_YAML):
    """EuRoC cam0 sensor.yaml -> dict(K, D, width, height).

    EuRoC distortion: 4 coefficients [k1, k2, p1, p2] (radial-tangential, plumb_bob).
    OpenCV expects 5 elements -> we append k3=0.
    """
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    fu, fv, cu, cv = d['intrinsics']
    K = np.array([[fu, 0.0, cu],
                  [0.0, fv,  cv],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(d.get('distortion_coefficients', []), dtype=np.float64)
    if len(dist) == 4:
        D = np.array([dist[0], dist[1], dist[2], dist[3], 0.0], dtype=np.float64)
    else:
        D = dist.astype(np.float64)
    w, h = d['resolution']
    return {'K': K, 'D': D, 'width': int(w), 'height': int(h)}


def load_extrinsic(yaml_path=CAM0_YAML):
    """EuRoC cam0 sensor.yaml's T_BS = body->sensor transform.

    EuRoC convention: body frame == IMU frame (imu0/sensor.yaml T_BS = I).
    So cam0's T_BS IS DIRECTLY T_imu_cam (p_imu = T_BS . p_cam). No inversion
    needed — none of the invert/direct ambiguity that STheReo has.

    Returns: 4x4 T_imu_cam.
    """
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    return np.array(d['T_BS']['data'], dtype=np.float64).reshape(4, 4)


def load_ground_truth(gt_csv=GT_CSV):
    """EuRoC state_groundtruth_estimate0/data.csv.

    Kolonlar: t [ns], p_xyz, q_wxyz, v_xyz, b_gyro_xyz, b_accel_xyz.
    Timestamp nanosaniyeden saniyeye çevrilir.
    """
    raw = np.loadtxt(gt_csv, delimiter=',')
    t = raw[:, 0]
    if t[0] > 1e12:
        t = t / 1e9
    return {
        't':  t,
        'p':  raw[:, 1:4].astype(np.float64),
        'q':  raw[:, 4:8].astype(np.float64),    # [qw, qx, qy, qz]
        'v':  raw[:, 8:11].astype(np.float64),
        'bg': raw[:, 11:14].astype(np.float64),
        'ba': raw[:, 14:17].astype(np.float64),
    }


# ----------------------------------------------------------------------
# IMU NOISE MODEL — ADIS16448 (imu0/sensor.yaml)
# ----------------------------------------------------------------------
IMU_GYRO_NOISE_DENSITY  = 1.6968e-04    # rad / s / sqrt(Hz)
IMU_ACCEL_NOISE_DENSITY = 2.0000e-03    # m / s² / sqrt(Hz)
IMU_GYRO_BIAS_RW        = 1.9393e-05    # rad / s² / sqrt(Hz)
IMU_ACCEL_BIAS_RW       = 3.0000e-03    # m / s³ / sqrt(Hz)

GRAVITY_MAGNITUDE = 9.81

ang_ran_walk_sqrt_hour = 0.66
ang_ran_walk_sqrt_sec = ang_ran_walk_sqrt_hour / np.sqrt(3600)
print(f'Gyro bias random walk (sqrt(Hz)): {ang_ran_walk_sqrt_sec:.2e} rad/s²/√Hz')


# ----------------------------------------------------------------------
# SEGMENT ANALYSIS — fixed thresholds (no auto-tuning); scene-scale dependent
# ----------------------------------------------------------------------
SEG_THRESHOLD_M    = 0.5    # tracking/diverged boundary on 4-DOF-aligned |err|
SEG_MIN_DURATION_S = 2.0     # shorter excursions/dips are transients
