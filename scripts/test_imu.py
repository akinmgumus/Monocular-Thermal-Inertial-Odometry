#!/usr/bin/env python3
"""Analyze gyro and acceleration norms for the first 5 seconds of xsens_imu.csv."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "/home/akin/tvio_ws/data/valley_evening/sensor_data/xsens_imu.csv"
WINDOW_SECONDS = 5.0

# Columns: timestamp, quat_x, quat_y, quat_z, quat_w, euler_x, euler_y, euler_z,
#          gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, mag_x, mag_y, mag_z
col_names = [
    "timestamp",
    "quat_x", "quat_y", "quat_z", "quat_w",
    "euler_x", "euler_y", "euler_z",
    "gyro_x", "gyro_y", "gyro_z",
    "acc_x", "acc_y", "acc_z",
    "mag_x", "mag_y", "mag_z",
]

df = pd.read_csv(CSV_PATH, header=None, names=col_names)

# Timestamps are in nanoseconds — convert to seconds relative to first sample
t0 = df["timestamp"].iloc[0]
df["t_sec"] = (df["timestamp"] - t0) / 1e9

mask = df["t_sec"] <= WINDOW_SECONDS
df5 = df[mask].copy()

print(f"Samples in first {WINDOW_SECONDS}s: {len(df5)}")
print(f"Time range: {df5['t_sec'].iloc[0]:.4f}s — {df5['t_sec'].iloc[-1]:.4f}s")

# Norms
df5["gyro_norm"] = np.sqrt(df5["gyro_x"]**2 + df5["gyro_y"]**2 + df5["gyro_z"]**2)
df5["acc_norm"]  = np.sqrt(df5["acc_x"]**2  + df5["acc_y"]**2  + df5["acc_z"]**2)

print("\n--- Gyro norm (rad/s) ---")
print(df5["gyro_norm"].describe())
print("\n--- Acc norm (m/s²) ---")
print(df5["acc_norm"].describe())

# Expected: gyro_norm ≈ 0 (stationary), acc_norm ≈ 9.81 (gravity only)
print(f"\nAcc norm mean: {df5['acc_norm'].mean():.4f} m/s²  (expected ~9.81 if stationary)")
print(f"Gyro norm mean: {df5['gyro_norm'].mean():.6f} rad/s  (expected ~0 if stationary)")

# Plot: 6 subplots (gyro x/y/z + acc x/y/z), each with mean dashed line
gyro_axes = ["gyro_x", "gyro_y", "gyro_z"]
acc_axes  = ["acc_x",  "acc_y",  "acc_z"]
gyro_colors = ["tab:red", "tab:green", "tab:blue"]
acc_colors  = ["tab:orange", "tab:purple", "tab:brown"]
gyro_labels = ["Gyro X (rad/s)", "Gyro Y (rad/s)", "Gyro Z (rad/s)"]
acc_labels  = ["Acc X (m/s²)",   "Acc Y (m/s²)",   "Acc Z (m/s²)"]

fig, axes = plt.subplots(6, 1, figsize=(13, 14), sharex=True)
fig.suptitle(f"IMU per-axis — First {WINDOW_SECONDS}s (valley_evening / xsens)", fontsize=12)

for i, (col, color, ylabel) in enumerate(zip(gyro_axes, gyro_colors, gyro_labels)):
    mean_val = df5[col].mean()
    axes[i].plot(df5["t_sec"], df5[col], color=color, linewidth=0.8, label=col)
    axes[i].axhline(mean_val, color="black", linestyle="--", linewidth=1,
                    label=f"mean = {mean_val:.6f} rad/s")
    axes[i].set_ylabel(ylabel, fontsize=8)
    axes[i].legend(fontsize=7, loc="upper right")
    axes[i].grid(True, alpha=0.4)

for i, (col, color, ylabel) in enumerate(zip(acc_axes, acc_colors, acc_labels)):
    mean_val = df5[col].mean()
    axes[3 + i].plot(df5["t_sec"], df5[col], color=color, linewidth=0.8, label=col)
    axes[3 + i].axhline(mean_val, color="black", linestyle="--", linewidth=1,
                        label=f"mean = {mean_val:.4f} m/s²")
    axes[3 + i].set_ylabel(ylabel, fontsize=8)
    axes[3 + i].legend(fontsize=7, loc="upper right")
    axes[3 + i].grid(True, alpha=0.4)

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
out_path = "/home/akin/tvio_ws/results/imu_first5s_analysis.png"
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")
plt.show()
