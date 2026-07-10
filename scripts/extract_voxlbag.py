"""Extract thermal_left, IMU, and PX4 odometry from a decoded VOXL ROS2 bag
into a flat on-disk layout compatible with thermal_vo.dataloader.

Output structure:
    <out_dir>/
    ├── cam0/
    │   ├── data/NNNNN.png      (8-bit PNG, 5-digit frame index)
    │   └── timestamps.txt      (one second timestamp per line, float)
    ├── imu0/
    │   └── data.csv            (t gx gy gz ax ay az, space-separated, seconds)
    └── gt/
        └── traj.txt            (TUM-style: t x y z qx qy qz qw, from /px4/odom)

The bag must be a ROS2 bag directory (with metadata.yaml + .db3 file). It is
expected to come from `ros2 bag record` with at least these topics:
  /ircam/decoded   (sensor_msgs/Image, mono8 from H264 decoder)
  /imu_apps        (sensor_msgs/Imu)
  /px4/odom        (nav_msgs/Odometry)

Usage:
    python scripts/extract_voxlbag.py \\
        --bag ~/voxlbag_data/decoded_first_2min_v3 \\
        --out ~/voxlbag_data/processed/run_v3
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


CAM_TOPIC = '/ircam/decoded'
IMU_TOPIC = '/imu_apps'
GT_TOPIC  = '/px4/odom'
GPS_TOPIC = '/fmu/out/vehicle_gps_position'

# Source of the px4_msgs/SensorGps .msg definition (so rosbags can deserialize
# it without a ROS2 environment). Adjust if your px4_msgs is cloned elsewhere.
PX4_MSGS_DIR = os.path.expanduser('~/aerial_tn_ws/src/px4_msgs/msg')


def image_msg_to_array(msg) -> np.ndarray:
    """Convert sensor_msgs/Image (mono8 or mono16) to a numpy array."""
    enc = msg.encoding
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ('mono8', '8UC1'):
        return buf.reshape(h, w)
    if enc in ('mono16', '16UC1'):
        return buf.view(np.uint16).reshape(h, w)
    raise ValueError(f'Unsupported image encoding: {enc}')


def header_stamp_to_seconds(stamp) -> float:
    """sensor_msgs Header.stamp -> float seconds (with nanosecond precision)."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True, type=Path,
                    help='Path to ROS2 bag directory (the one with metadata.yaml).')
    ap.add_argument('--out', required=True, type=Path,
                    help='Output directory (will be created).')
    ap.add_argument('--cam-topic', default=CAM_TOPIC)
    ap.add_argument('--imu-topic', default=IMU_TOPIC)
    ap.add_argument('--gt-topic',  default=GT_TOPIC)
    ap.add_argument('--gps-topic', default=GPS_TOPIC)
    ap.add_argument('--no-gps',    action='store_true',
                    help='Skip GPS extraction (default: extract if topic present).')
    args = ap.parse_args()

    bag_path = args.bag.expanduser().resolve()
    out_dir  = args.out.expanduser().resolve()

    cam_dir = out_dir / 'cam0' / 'data'
    imu_dir = out_dir / 'imu0'
    gt_dir  = out_dir / 'gt'
    gps_dir = out_dir / 'gps'
    cam_dir.mkdir(parents=True, exist_ok=True)
    imu_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    typestore = get_typestore(Stores.LATEST)

    # Register px4_msgs/SensorGps so we can deserialize GPS messages without
    # a ROS2 environment. Skip silently if the .msg file is absent.
    if not args.no_gps:
        try:
            from rosbags.typesys import get_types_from_msg
            sensor_gps_path = Path(PX4_MSGS_DIR) / 'SensorGps.msg'
            if sensor_gps_path.exists():
                types = get_types_from_msg(sensor_gps_path.read_text(),
                                           'px4_msgs/msg/SensorGps')
                typestore.register(types)
                print(f'Registered px4_msgs/msg/SensorGps from {sensor_gps_path}')
            else:
                print(f'[warn] px4_msgs/msg/SensorGps.msg not found at {sensor_gps_path}'
                      ' — GPS extraction will be skipped.')
                args.no_gps = True
        except Exception as e:
            print(f'[warn] Could not register px4_msgs types ({e}); GPS skipped.')
            args.no_gps = True

    cam_timestamps = []   # seconds, in extraction order
    imu_rows = []         # [t_s, gx, gy, gz, ax, ay, az]
    gt_rows  = []         # [t_s, x, y, z, qx, qy, qz, qw]
    gps_rows = []         # [t_s, lat_deg, lon_deg, alt_m, fix_type, eph, epv, sats]

    with Reader(bag_path) as r:
        cam_conns = [c for c in r.connections if c.topic == args.cam_topic]
        imu_conns = [c for c in r.connections if c.topic == args.imu_topic]
        gt_conns  = [c for c in r.connections if c.topic == args.gt_topic]
        gps_conns = ([c for c in r.connections if c.topic == args.gps_topic]
                     if not args.no_gps else [])
        if not cam_conns:
            sys.exit(f'No connection for camera topic {args.cam_topic}')
        if not imu_conns:
            sys.exit(f'No connection for IMU topic {args.imu_topic}')
        if not gt_conns:
            print(f'[warn] no connection for GT topic {args.gt_topic} — skipping GT')
        if not args.no_gps and not gps_conns:
            print(f'[warn] no connection for GPS topic {args.gps_topic} — skipping GPS')

        cam_total = sum(c.msgcount for c in cam_conns)
        imu_total = sum(c.msgcount for c in imu_conns)
        gt_total  = sum(c.msgcount for c in gt_conns)
        gps_total = sum(c.msgcount for c in gps_conns)
        print(f'Found  {cam_total} camera, {imu_total} IMU, '
              f'{gt_total} odom, {gps_total} GPS messages.')
        print(f'Writing to {out_dir}')

        all_conns = cam_conns + imu_conns + gt_conns + gps_conns
        for conn, t_ns, raw in r.messages(connections=all_conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)

            if conn.topic == args.cam_topic:
                idx = len(cam_timestamps)
                img = image_msg_to_array(msg)
                cv2.imwrite(str(cam_dir / f'{idx:05d}.png'), img)
                cam_timestamps.append(header_stamp_to_seconds(msg.header.stamp))
                if (idx + 1) % 500 == 0:
                    print(f'  cam: wrote {idx + 1}/{cam_total}')

            elif conn.topic == args.imu_topic:
                imu_rows.append([
                    header_stamp_to_seconds(msg.header.stamp),
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ])

            elif conn.topic == args.gt_topic:
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                gt_rows.append([
                    header_stamp_to_seconds(msg.header.stamp),
                    p.x, p.y, p.z,
                    q.x, q.y, q.z, q.w,
                ])

            elif conn.topic == args.gps_topic:
                # px4_msgs/SensorGps: lat/lon in 1e-7 deg, alt in mm above MSL.
                # PX4 .timestamp is microseconds since system start — same epoch
                # as IMU header.stamp on this platform; we convert to seconds.
                gps_rows.append([
                    msg.timestamp * 1e-6,
                    msg.lat * 1e-7,
                    msg.lon * 1e-7,
                    msg.alt * 1e-3,
                    float(msg.fix_type),
                    msg.eph,
                    msg.epv,
                    float(msg.satellites_used),
                ])

    # ---- Write outputs ----
    ts_path = out_dir / 'cam0' / 'timestamps.txt'
    np.savetxt(ts_path, np.asarray(cam_timestamps, dtype=np.float64), fmt='%.9f')

    imu_arr = np.asarray(imu_rows, dtype=np.float64)
    imu_arr = imu_arr[np.argsort(imu_arr[:, 0])]
    imu_path = imu_dir / 'data.csv'
    np.savetxt(imu_path, imu_arr, fmt='%.9f')

    if gt_rows:
        gt_arr = np.asarray(gt_rows, dtype=np.float64)
        gt_arr = gt_arr[np.argsort(gt_arr[:, 0])]
        gt_path = gt_dir / 'traj.txt'
        np.savetxt(gt_path, gt_arr, fmt='%.9f',
                   header='timestamp x y z q_x q_y q_z q_w (TUM)', comments='# ')

    if gps_rows:
        gps_dir.mkdir(parents=True, exist_ok=True)
        gps_arr = np.asarray(gps_rows, dtype=np.float64)
        gps_arr = gps_arr[np.argsort(gps_arr[:, 0])]
        gps_path = gps_dir / 'data.csv'
        np.savetxt(gps_path, gps_arr, fmt='%.9f',
                   header='timestamp lat_deg lon_deg alt_m_msl fix_type eph_m epv_m sats',
                   comments='# ')

    print()
    print(f'Done. {len(cam_timestamps)} images, {len(imu_rows)} IMU samples, '
          f'{len(gt_rows)} odom poses, {len(gps_rows)} GPS fixes.')
    print(f'  cam timestamps : {ts_path}')
    print(f'  IMU            : {imu_path}')
    if gt_rows:
        print(f'  GT trajectory  : {gt_path}')
    if gps_rows:
        print(f'  GPS            : {gps_path}')


if __name__ == '__main__':
    main()
