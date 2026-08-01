"""Extract ROVTIO alt1/alt2/lt* recordings into a flat on-disk layout
compatible with mTIO.dataloader.

The ROVTIO dataset ships as 10 split ROS1 .bag files for a single recording
(`m100_charlie_2021-06-09-23-XX-XX_N.bag`). This script reads them in
chronological order as one logical stream and writes:

    <out_dir>/
    ├── cam0/
    │   ├── data/NNNNN.png      (16-bit PNG, 5-digit frame index)
    │   └── timestamps.txt      (one second timestamp per line, float)
    ├── imu0/
    │   └── data.csv            (t gx gy gz ax ay az, seconds)
    └── gt/
        └── traj.txt            (TUM: t x y z qx qy qz qw, from Vicon)

Topics extracted:
    /tau_nodelet/thermal_image  — FLIR Tau2 thermal, 16UC1, 640x512, ~25 Hz
    /vn100/imu                  — VectorNav VN-100, ~168 Hz
    /vicon/charlie/charlie      — Vicon motion capture GT (TransformStamped, ~6 Hz)

Usage:
    python scripts/extract_rovtio_bag.py \\
        --bag-dir ~/Downloads/alt1 \\
        --out ~/rovtio_data/processed/alt1
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


CAM_TOPIC = '/tau_nodelet/thermal_image'
IMU_TOPIC = '/vn100/imu'
GT_TOPIC  = '/vicon/charlie/charlie'


def image_msg_to_array(msg) -> np.ndarray:
    """Convert sensor_msgs/Image (16UC1) to uint16 numpy array."""
    enc = msg.encoding
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ('16UC1', 'mono16'):
        return buf.view(np.uint16).reshape(h, w)
    if enc in ('mono8', '8UC1'):
        return buf.reshape(h, w)
    raise ValueError(f'Unsupported image encoding: {enc}')


def header_stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag-dir', required=True, type=Path,
                    help='Directory containing the split .bag files for one recording.')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--cam-topic', default=CAM_TOPIC)
    ap.add_argument('--imu-topic', default=IMU_TOPIC)
    ap.add_argument('--gt-topic',  default=GT_TOPIC)
    args = ap.parse_args()

    bag_dir = args.bag_dir.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()

    # Find and sort split bag files (they're suffixed _0, _1, ..., _9).
    bags = sorted(bag_dir.glob('*.bag'),
                  key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    if not bags:
        sys.exit(f'No .bag files found in {bag_dir}')
    print(f'Found {len(bags)} bag fragments:')
    for b in bags:
        print(f'  {b.name}')

    cam_dir = out_dir / 'cam0' / 'data'
    imu_dir = out_dir / 'imu0'
    gt_dir  = out_dir / 'gt'
    cam_dir.mkdir(parents=True, exist_ok=True)
    imu_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    typestore = get_typestore(Stores.ROS1_NOETIC)

    cam_timestamps = []
    imu_rows = []
    gt_rows  = []

    for bag in bags:
        print(f'\n→ {bag.name}')
        with Reader(bag) as r:
            cam_conns = [c for c in r.connections if c.topic == args.cam_topic]
            imu_conns = [c for c in r.connections if c.topic == args.imu_topic]
            gt_conns  = [c for c in r.connections if c.topic == args.gt_topic]
            if not cam_conns:
                print(f'  [warn] {args.cam_topic} absent in this fragment')
            if not imu_conns:
                print(f'  [warn] {args.imu_topic} absent in this fragment')
            if not gt_conns:
                print(f'  [warn] {args.gt_topic} absent in this fragment')

            cam_in_frag = imu_in_frag = gt_in_frag = 0
            for conn, t_ns, raw in r.messages(connections=cam_conns + imu_conns + gt_conns):
                msg = typestore.deserialize_ros1(raw, conn.msgtype)

                if conn.topic == args.cam_topic:
                    idx = len(cam_timestamps)
                    img = image_msg_to_array(msg)
                    cv2.imwrite(str(cam_dir / f'{idx:05d}.png'), img)
                    cam_timestamps.append(header_stamp_to_seconds(msg.header.stamp))
                    cam_in_frag += 1

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
                    imu_in_frag += 1

                elif conn.topic == args.gt_topic:
                    p = msg.transform.translation
                    q = msg.transform.rotation
                    gt_rows.append([
                        header_stamp_to_seconds(msg.header.stamp),
                        p.x, p.y, p.z,
                        q.x, q.y, q.z, q.w,
                    ])
                    gt_in_frag += 1

            print(f'  + {cam_in_frag} cam, {imu_in_frag} imu, {gt_in_frag} gt messages')

    # Write outputs (seconds, float, sorted by time).
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

    print()
    print(f'Done. {len(cam_timestamps)} images, {len(imu_rows)} IMU samples, '
          f'{len(gt_rows)} GT poses.')
    print(f'  cam timestamps : {ts_path}')
    print(f'  IMU            : {imu_path}')
    if gt_rows:
        print(f'  GT (Vicon)     : {gt_path}')


if __name__ == '__main__':
    main()