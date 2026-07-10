"""Extract thermal_left images and IMU from a FIReStereo rosbag into a flat
on-disk layout compatible with thermal_vo.dataloader.

Output structure:
    <out_dir>/
    ├── cam0/
    │   ├── data/NNNNN.png      (16-bit PNG, 5-digit frame index)
    │   └── timestamps.txt      (one nanosecond timestamp per line)
    └── imu0/
        └── data.csv            (t_ns, gx, gy, gz, ax, ay, az, space-separated)

Usage:
    python scripts/extract_firestereo_bag.py \\
        --bag ~/FIReStereo_data/rosbags_ros2/frick_1 \\
        --out ~/FIReStereo_data/processed/frick_1

The bag path may be either a ROS1 .bag file or a ROS2 bag directory.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def open_reader(bag_path: Path):
    """Return an open rosbags Reader for either ROS1 .bag or ROS2 bag directory."""
    if bag_path.is_file() and bag_path.suffix == '.bag':
        from rosbags.rosbag1 import Reader
        return Reader(bag_path), 'ros1'
    if bag_path.is_dir():
        from rosbags.rosbag2 import Reader
        return Reader(bag_path), 'ros2'
    raise FileNotFoundError(f'Bag not found or unsupported type: {bag_path}')


def deserialize(typestore_kind, raw, msgtype, typestore):
    """Deserialize one raw message using the right backend for the bag format."""
    if typestore_kind == 'ros1':
        return typestore.deserialize_ros1(raw, msgtype)
    return typestore.deserialize_cdr(raw, msgtype)


def image_msg_to_array(msg) -> np.ndarray:
    """Convert sensor_msgs/Image (mono16 or mono8) to a numpy array."""
    enc = msg.encoding
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ('mono16', '16UC1'):
        return buf.view(np.uint16).reshape(h, w)
    if enc in ('mono8', '8UC1'):
        return buf.reshape(h, w)
    raise ValueError(f'Unsupported image encoding: {enc}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True, type=Path,
                    help='Path to ROS1 .bag file OR ROS2 bag directory.')
    ap.add_argument('--out', required=True, type=Path,
                    help='Output directory (will be created).')
    ap.add_argument('--cam-topic', default='/thermal_left/image')
    ap.add_argument('--imu-topic', default='/imu/data')
    args = ap.parse_args()

    bag_path = args.bag.expanduser().resolve()
    out_dir  = args.out.expanduser().resolve()

    cam_dir = out_dir / 'cam0' / 'data'
    imu_dir = out_dir / 'imu0'
    cam_dir.mkdir(parents=True, exist_ok=True)
    imu_dir.mkdir(parents=True, exist_ok=True)

    from rosbags.typesys import Stores, get_typestore

    reader, kind = open_reader(bag_path)
    typestore = get_typestore(Stores.ROS1_NOETIC if kind == 'ros1' else Stores.LATEST)

    cam_timestamps = []   # ns, in extraction order
    imu_rows = []         # rows of [t_ns, gx, gy, gz, ax, ay, az]

    with reader as r:
        cam_conns = [c for c in r.connections if c.topic == args.cam_topic]
        imu_conns = [c for c in r.connections if c.topic == args.imu_topic]
        if not cam_conns:
            sys.exit(f'No connection for camera topic {args.cam_topic}')
        if not imu_conns:
            sys.exit(f'No connection for IMU topic {args.imu_topic}')

        cam_total = sum(c.msgcount for c in cam_conns)
        imu_total = sum(c.msgcount for c in imu_conns)
        print(f'Found {cam_total} camera messages and {imu_total} IMU messages.')
        print(f'Writing to {out_dir}')

        for conn, t_ns, raw in r.messages(connections=cam_conns + imu_conns):
            msg = deserialize(kind, raw, conn.msgtype, typestore)

            if conn.topic == args.cam_topic:
                idx = len(cam_timestamps)
                img = image_msg_to_array(msg)
                cv2.imwrite(str(cam_dir / f'{idx:05d}.png'), img)
                cam_timestamps.append(t_ns)
                if (idx + 1) % 500 == 0:
                    print(f'  cam: wrote {idx + 1}/{cam_total}')
            else:
                imu_rows.append([
                    t_ns,
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ])

    # Write timestamps.txt — seconds (float). Using float64 with explicit ns→s
    # conversion preserves nanosecond precision and matches the convention
    # already used by config_firestereo.load_ground_truth().
    ts_path = out_dir / 'cam0' / 'timestamps.txt'
    ts_sec = np.array(cam_timestamps, dtype=np.int64).astype(np.float64) / 1e9
    np.savetxt(ts_path, ts_sec, fmt='%.9f')

    # Write IMU CSV (space-separated, no header — matches IMULoader defaults).
    # Convert ns timestamps to seconds here, with the same precision care.
    t_ns = np.array([r[0] for r in imu_rows], dtype=np.int64)
    rest = np.array([r[1:] for r in imu_rows], dtype=np.float64)
    order = np.argsort(t_ns)
    t_s = t_ns[order].astype(np.float64) / 1e9
    imu_out = np.column_stack([t_s, rest[order]])
    imu_path = imu_dir / 'data.csv'
    np.savetxt(imu_path, imu_out, fmt='%.9f')

    print()
    print(f'Done. Wrote {len(cam_timestamps)} images, {len(imu_rows)} IMU samples.')
    print(f'  cam timestamps : {ts_path}')
    print(f'  IMU            : {imu_path}')


if __name__ == '__main__':
    main()