"""Dataset paths, calibration loaders, and IMU noise constants for VOXL bag.

VOXL Starling 2 Max V2 (C28 config) drone. Recorded by ModalAI VOXL platform,
H.264-encoded thermal stream + 472 Hz IMU + PX4 EKF2 odometry. Decoded by
AerialTN_utility (decoder_launch.py) and re-recorded with ros2 bag record,
then extracted to flat layout by scripts/extract_voxlbag.py.

Important caveats — read before tuning:

  1) IMU noise constants below are ALLAN-CALIBRATED values shared by the
     dataset owner (~20 hour static recording, processed via Kalibr).
     File of origin: kalibr_imu_chain.yaml (default factory calibration
     for the VOXL Starling 2 Max C28 platform). Values are already
     inflated by the supplier (~5x on white noise, ~10x on random walk)
     to account for unmodelled effects (vibration, temperature drift).
     IMU model: TDK ICM-42688-P on the ModalAI VOXL2 board.

  2) No static segment exists in the available 2-minute clip — drone is
     already maneuvering at t=0 (gyro >0.1 rad/s throughout). Bias init
     therefore CANNOT use gravity-aligned-from-static method as in
     FIReStereo. Instead, the test script derives R_init from /px4/odom's
     first pose and computes ba_init from the quasi-static window
     t in [INIT_QUASI_STATIC_T0, INIT_QUASI_STATIC_T1].

  3) Thermal is mono8 (H.264-decoded). Original sensor is 16-bit FLIR
     Boson+; H.264 perceptual coding is destructive for thermal gradient
     information. Pipeline works but quality < raw thermal.

  4) Rolling shutter (33 ms readout) not modeled. Expect degraded VIO under
     high-yaw maneuvers.

  5) Camera-IMU extrinsics are taken from extrinsics.conf entry for drone
     'D0012_Starling_2_Max_V2_C28_no_tof'. The recording bag may have come
     from a different unit of the same platform — drone-to-drone variation
     is typically <1 cm but exists. Verify with dataset owners when possible.
"""

import os
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


# ----------------------------------------------------------------------
# DATASET LAYOUT
# ----------------------------------------------------------------------
DATA_ROOT     = os.path.expanduser('~/voxlbag_data')
PROCESSED_DIR = os.path.join(DATA_ROOT, 'processed', 'run_v3')
BAG_PATH      = os.path.join(DATA_ROOT, 'decoded_first_2min_v3')   # for reference

CAM0_DIR      = os.path.join(PROCESSED_DIR, 'cam0', 'data')
CAM0_TS       = os.path.join(PROCESSED_DIR, 'cam0', 'timestamps.txt')
IMU_CSV       = os.path.join(PROCESSED_DIR, 'imu0', 'data.csv')
GT_TRAJ       = os.path.join(PROCESSED_DIR, 'gt', 'traj.txt')
GPS_CSV       = os.path.join(PROCESSED_DIR, 'gps', 'data.csv')

# AerialTN_utility supplied calibration (cloned ROS2 workspace). Note the repo
# was reorganized after our initial clone: the calibration folder is now
# 'camera_calibration/' (was 'camera_calibs_for_voxl/'), and the legacy
# drone_config.yaml was removed in favor of split per-purpose files.
AERIALTN_DIR  = os.path.expanduser('~/aerial_tn_ws/src/AerialTN_utility/voxl_data')
CAM_YAML      = os.path.join(AERIALTN_DIR, 'camera_calibration', 'opencv_ircam_intrinsics.yml')
EXTRINSICS_V1 = os.path.join(AERIALTN_DIR, 'configuration_files', 'extrinsics_v1.conf')
IMU_KALIBR    = os.path.join(AERIALTN_DIR, 'imu_calibration', 'kalibr_imu_chain.yaml')
IMU_SERVER_CAL = os.path.join(AERIALTN_DIR, 'imu_calibration', 'voxl-imu-server.cal')

# Bag topic names (for reference; not used directly by the pipeline).
TOPIC_THERMAL = '/ircam/decoded'
TOPIC_IMU     = '/imu_apps'
TOPIC_ODOM    = '/px4/odom'


# ----------------------------------------------------------------------
# CALIBRATION LOADERS
# ----------------------------------------------------------------------
def load_camera_intrinsics(yaml_path=CAM_YAML):
    """Load thermal (ircam) intrinsics from OpenCV YAML.

    The file is in OpenCV's '%YAML:1.0' format with !!opencv-matrix tags.
    A small custom loader handles those tags.

    Returns: dict(K, D, width, height).
    """
    def opencv_matrix_constructor(loader, node):
        m = loader.construct_mapping(node, deep=True)
        return np.array(m['data']).reshape((m['rows'], m['cols']))
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:opencv-matrix',
                                    opencv_matrix_constructor)
    # OpenCV emits a leading '%YAML:1.0' directive that PyYAML rejects;
    # strip it before parsing.
    with open(yaml_path) as f:
        raw = f.read()
    if raw.startswith('%YAML'):
        raw = '\n'.join(raw.splitlines()[1:])
    d = yaml.safe_load(raw)
    K = np.asarray(d['M'], dtype=np.float64)
    D = np.asarray(d['D'], dtype=np.float64).flatten()
    if len(D) == 4:
        D = np.append(D, 0.0)
    return {'K': K, 'D': D, 'width': int(d['width']), 'height': int(d['height'])}


def load_extrinsic_imu_cam(yaml_path=None):
    """Return T_imu_cam (4x4) for the thermal camera.

    Source: extrinsics_v1.conf (AerialTN_utility/voxl_data, post-update),
    entry 'ircam'. NOTE: the legacy 'lepton0_raw' entry in the old
    extrinsics.conf had a different rotation (-90° vs +90° about Z) and
    a different translation; using it caused Gauss-Newton triangulation
    to diverge on every track. Confirmed with dataset owner that the
    new v1 values below are correct for the bag we're using.

        parent = imu_apps
        child  = ircam
        T_child_wrt_parent  = [-0.11, 0.0, 0.036]  meters
        RPY_parent_to_child = [0, 0, +90]          degrees (intrinsic XYZ)

    Convention: p_imu = T_imu_cam * p_cam.
    """
    T = np.array([-0.11, 0.0, 0.036], dtype=np.float64)
    R_mat = R.from_euler('XYZ', [0.0, 0.0, 90.0], degrees=True).as_matrix()
    T_imu_cam = np.eye(4)
    T_imu_cam[:3, :3] = R_mat
    T_imu_cam[:3, 3]  = T
    return T_imu_cam


def load_timeshift_imu_cam():
    """IMU-camera time offset in seconds.

    The td=0.015 s in drone_config.yaml is documented by the dataset owner
    as a placeholder default — "Assume the time offset to be 0."
    """
    return 0.0


def _wgs84_to_ecef(lat_deg, lon_deg, alt_m):
    """WGS84 geodetic → ECEF Cartesian (meters)."""
    a  = 6378137.0
    e2 = 6.69437999014e-3
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1.0 - e2) + alt_m) * np.sin(lat)
    return x, y, z


def _ecef_to_enu(x, y, z, lat0_deg, lon0_deg, alt0_m):
    """ECEF → local ENU centered at (lat0, lon0, alt0)."""
    x0, y0, z0 = _wgs84_to_ecef(lat0_deg, lon0_deg, alt0_m)
    dx, dy, dz = x - x0, y - y0, z - z0
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    sl, cl = np.sin(lat0), np.cos(lat0)
    so, co = np.sin(lon0), np.cos(lon0)
    east  = -so * dx + co * dy
    north = -sl * co * dx - sl * so * dy + cl * dz
    up    =  cl * co * dx + cl * so * dy + sl * dz
    return np.stack([east, north, up], axis=-1)


def load_gps(gps_path=GPS_CSV):
    """Load raw GPS and convert WGS84 (lat/lon/alt) to local ENU.

    Origin is set at the first 3D-fix sample. Output ENU positions in meters.
    This is an INDEPENDENT reference (does not share IMU with our pipeline),
    but has ~1.8 m horizontal noise → useful for ATE, NOT for short-window RPE.

    Returns: dict with t [s], p_enu [Nx3, east-north-up meters],
             eph [N, horizontal accuracy m], epv [N, vertical accuracy m],
             sats [N], origin {lat, lon, alt}.
    Only rows with fix_type >= 3 (3D fix) are kept.
    """
    raw = np.loadtxt(gps_path, comments='#')
    # cols: t lat lon alt fix_type eph epv sats
    mask = raw[:, 4] >= 3
    raw = raw[mask]
    t   = raw[:, 0]
    lat = raw[:, 1]; lon = raw[:, 2]; alt = raw[:, 3]
    eph = raw[:, 5]; epv = raw[:, 6]; sats = raw[:, 7]

    x, y, z = _wgs84_to_ecef(lat, lon, alt)
    enu = _ecef_to_enu(x, y, z, lat[0], lon[0], alt[0])
    return {
        't':    t,
        'p_enu': enu,                     # [N, 3] east north up, m
        'eph':  eph,
        'epv':  epv,
        'sats': sats,
        'origin': {'lat': lat[0], 'lon': lon[0], 'alt': alt[0]},
    }


def load_ground_truth(gt_path=GT_TRAJ):
    """Load PX4 EKF2 odometry trajectory (decoded via AerialTN_utility).

    Format (TUM-style): '# timestamp x y z q_x q_y q_z q_w'

    Note: this is NOT motion-capture ground truth. It is PX4's EKF2 fusion
    output (GPS + IMU + magnetometer + barometer). For this dataset the same
    IMU feeds both /imu_apps (raw, used by us) and EKF2 (used here as GT),
    so the comparison is partially circular — drift in our IMU integration
    will be partly mirrored by EKF2. For an independent reference, use the
    raw GPS stream (/fmu/out/vehicle_gps_position) — noisier but independent.

    Returns: dict with t [s], p [Nx3], q [Nx4 in qx qy qz qw order].
    """
    raw = np.loadtxt(gt_path, comments='#')
    return {
        't': raw[:, 0].astype(np.float64),
        'p': raw[:, 1:4].astype(np.float64),
        'q': raw[:, 4:8].astype(np.float64),     # qx qy qz qw
    }


# ----------------------------------------------------------------------
# IMU NOISE MODEL — Allan-calibrated TDK ICM-42688-P (Starling 2 Max C28)
# ----------------------------------------------------------------------
# Source: kalibr_imu_chain.yaml shared by dataset owner. Allan-derived from
# ~20 hour static recording, inflated by supplier (~5x white noise, ~10x bias
# random walk) for unmodelled vibration / temperature effects.
#
# Raw Allan reference (what the .Conf file holds, NOT used directly):
#   accel_noise_density_raw = 3.895e-3   →  inflated 2.314e-2  ← used
#   accel_random_walk_raw   = 5.538e-5   →  inflated 2.382e-3  ← used
#   gyro_noise_density_raw  = 1.399e-4   →  inflated 9.713e-4  ← used
#   gyro_random_walk_raw    = 4.119e-7   →  inflated 6.866e-4  ← used
#
# History:
#   first VINS-Mono placeholder values (σ_g=0.05, σ_a=0.2, σ_bg=4e-3, σ_ba=1e-2)
#   then datasheet-MEMS estimate    (σ_g=1e-3, σ_a=5e-3, σ_bg=5e-5, σ_ba=1e-3)
#   now Allan-calibrated from owner — what's used below.
IMU_GYRO_NOISE_DENSITY  = 9.713e-04   # rad / s / sqrt(Hz)
IMU_ACCEL_NOISE_DENSITY = 2.314e-02   # m / s^2 / sqrt(Hz)
IMU_GYRO_BIAS_RW        = 6.866e-04   # rad / s^2 / sqrt(Hz)
IMU_ACCEL_BIAS_RW       = 2.382e-03   # m / s^3 / sqrt(Hz)


# ----------------------------------------------------------------------
# INITIALIZATION — quasi-static window for accel-bias derivation
# ----------------------------------------------------------------------
# Inspection of run_v3 IMU showed no truly static segment, but a quasi-static
# window in the 2.5-6.0 s range where |a| ~ 9.7 m/s² and gyro < 0.15 rad/s.
# The test script uses this window to derive ba_init = a_mean - R_init^T @ g,
# with R_init coming from /px4/odom's first pose.
INIT_QUASI_STATIC_T0 = 2.5     # seconds from first cam frame
INIT_QUASI_STATIC_T1 = 6.0

IMU_INIT_BIAS_GYRO  = np.array([ 0.00126, -0.00183,  0.00248])
IMU_INIT_BIAS_ACCEL = np.array([-0.01066,  0.03121, -0.07132])


GRAVITY_MAGNITUDE = 9.80        # VOXL drone_config uses 9.80, not 9.81


# Sampling rates (for sanity checks).
IMU_RATE_HZ     = 472.0
THERMAL_RATE_HZ =  30.0

# ----------------------------------------------------------------------
# SEGMENT ANALYSIS — fixed thresholds (no auto-tuning); scene-scale dependent
# ----------------------------------------------------------------------
SEG_THRESHOLD_M    = 5.0    # tracking/diverged boundary on 4-DOF-aligned |err|
SEG_MIN_DURATION_S = 2.0     # shorter excursions/dips are transients
