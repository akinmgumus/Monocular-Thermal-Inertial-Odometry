#!/usr/bin/env python3
"""
IMU-only dead reckoning trajectory with live animation.
Gyro → SO(3) rotation, Accel → velocity → position.
Shows drift explosion visually.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D

# ── Load ────────────────────────────────────────────────────────────────────
CSV = "/home/akin/tvio_ws/data/hawkins_4_imu.csv"
df = pd.read_csv(CSV, comment="#",
                 names=["timestamp", "wx", "wy", "wz", "ax", "ay", "az"])
df = df.astype(float).reset_index(drop=True)
N = len(df)
print(f"Loaded {N} IMU samples  (~{N/200:.1f}s at 200Hz)")

# ── Dead Reckoning ───────────────────────────────────────────────────────────
# Gravity vector in world frame.
# At rest az ≈ +9.81, so sensor measures reaction to gravity upward.
# In body frame: g_body ≈ [0, 0, +9.81].  After rotating to world: subtract same.
G_WORLD = np.array([0.0, 0.0, 9.81])

ts  = df["timestamp"].values
oms = df[["wx", "wy", "wz"]].values   # angular velocity  (rad/s)
acc = df[["ax", "ay", "az"]].values   # specific force    (m/s²)

positions  = np.zeros((N, 3))
velocities = np.zeros((N, 3))
R = np.eye(3)
v = np.zeros(3)
p = np.zeros(3)

for i in range(1, N):
    dt = ts[i] - ts[i - 1]
    if dt <= 0 or dt > 0.1:          # skip bad timestamps
        positions[i]  = positions[i - 1]
        velocities[i] = velocities[i - 1]
        continue

    # Rotation update  R ← R · Exp(ω·dt)
    R = R @ Rotation.from_rotvec(oms[i] * dt).as_matrix()

    # Remove gravity, get world-frame acceleration
    a_world = R @ acc[i] - G_WORLD
    a_world[np.abs(a_world) < 0.5] = 0

    # Euler integration
    v = v + a_world * dt
    p = p + v * dt

    positions[i]  = p
    velocities[i] = v

vel_mag = np.linalg.norm(velocities, axis=1)   # m/s magnitude

# ── Find "explosion" point ───────────────────────────────────────────────────
EXPLOSION_THRESH = 5.0   # m/s  — robot shouldn't go faster than this indoors
explode_idx = np.argmax(vel_mag > EXPLOSION_THRESH)
explode_idx = explode_idx if explode_idx > 0 else N - 1
explode_t   = ts[explode_idx] - ts[0]
print(f"Velocity > {EXPLOSION_THRESH} m/s first at sample {explode_idx}  "
      f"(t = {explode_t:.2f}s)")

# ── Animation setup ──────────────────────────────────────────────────────────
STEP   = 50      # draw every Nth sample  (tune for speed)
frames = np.arange(0, N, STEP)

fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor("#0d0d0d")

ax3d  = fig.add_subplot(2, 2, (1, 3), projection="3d")   # left column
ax_vm = fig.add_subplot(2, 2, 2)                          # top-right
ax_xy = fig.add_subplot(2, 2, 4)                          # bottom-right

for ax in [ax_vm, ax_xy]:
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

ax3d.set_facecolor("#111111")
ax3d.tick_params(colors="white")
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False

fig.suptitle("IMU Dead Reckoning — Live Drift", color="white", fontsize=13)

# ── Static reference lines ───────────────────────────────────────────────────
rel_t = ts - ts[0]
ax_vm.plot(rel_t, vel_mag, color="#333333", lw=0.5, zorder=1)
ax_vm.axvline(explode_t, color="#ff4444", lw=1.2, ls="--", label=f"drift explodes ~{explode_t:.1f}s")
ax_vm.axhline(EXPLOSION_THRESH, color="#ff8800", lw=0.8, ls=":")
ax_vm.set_xlabel("Time (s)")
ax_vm.set_ylabel("|v| (m/s)")
ax_vm.set_title("Velocity magnitude", color="white")
ax_vm.legend(fontsize=8, facecolor="#222", labelcolor="white")

ax_xy.set_xlabel("X (m)")
ax_xy.set_ylabel("Y (m)")
ax_xy.set_title("Top-down (XY)", color="white")
ax_xy.set_aspect("equal", adjustable="datalim")

# ── Animated artists ─────────────────────────────────────────────────────────
line3d,   = ax3d.plot([], [], [], color="#00ccff", lw=0.8)
dot3d,    = ax3d.plot([], [], [], "o", color="#ffffff", ms=4)
line_vm,  = ax_vm.plot([], [], color="#00ccff", lw=1.2, zorder=2)
vline_vm  = ax_vm.axvline(0, color="white", lw=0.8, ls="-", zorder=3)
line_xy,  = ax_xy.plot([], [], color="#00ccff", lw=0.8)
dot_xy,   = ax_xy.plot([], [], "o", color="#ffffff", ms=4)
start_xy, = ax_xy.plot(positions[0, 0], positions[0, 1], "^",
                       color="#44ff88", ms=6, label="start")
ax_xy.legend(fontsize=8, facecolor="#222", labelcolor="white")

# Axis limits (clip to sane range to avoid matplotlib blowing up on huge drift)
MAX_RANGE = 50.0   # metres — beyond this drift is already insane to look at
clip = np.abs(positions) < MAX_RANGE
valid = np.where(clip.all(axis=1))[0]
if len(valid):
    lo = positions[valid].min(axis=0)
    hi = positions[valid].max(axis=0)
    pad = max((hi - lo).max() * 0.1, 0.5)
    ax3d.set_xlim(lo[0] - pad, hi[0] + pad)
    ax3d.set_ylim(lo[1] - pad, hi[1] + pad)
    ax3d.set_zlim(lo[2] - pad, hi[2] + pad)

ax3d.set_xlabel("X", color="white", labelpad=2)
ax3d.set_ylabel("Y", color="white", labelpad=2)
ax3d.set_zlabel("Z", color="white", labelpad=2)
ax3d.set_title("3-D trajectory", color="white")

# color segment: normal = cyan, post-explosion = red
def seg_color(idx):
    return "#ff4444" if idx >= explode_idx else "#00ccff"

def init():
    line3d.set_data([], [])
    line3d.set_3d_properties([])
    dot3d.set_data([], [])
    dot3d.set_3d_properties([])
    line_vm.set_data([], [])
    line_xy.set_data([], [])
    dot_xy.set_data([], [])
    return line3d, dot3d, line_vm, vline_vm, line_xy, dot_xy

def update(frame_idx):
    i = frames[frame_idx]
    xs, ys, zs = positions[:i+1, 0], positions[:i+1, 1], positions[:i+1, 2]

    # Clamp for 3D view
    mask = (np.abs(xs) < MAX_RANGE) & (np.abs(ys) < MAX_RANGE) & (np.abs(zs) < MAX_RANGE)
    xc, yc, zc = xs[mask], ys[mask], zs[mask]

    line3d.set_data(xc, yc)
    line3d.set_3d_properties(zc)
    line3d.set_color(seg_color(i))

    if len(xc):
        dot3d.set_data([xc[-1]], [yc[-1]])
        dot3d.set_3d_properties([zc[-1]])

    t_now = rel_t[i]
    line_vm.set_data(rel_t[:i+1], vel_mag[:i+1])
    vline_vm.set_xdata([t_now, t_now])

    line_xy.set_data(xs[mask], ys[mask])
    line_xy.set_color(seg_color(i))
    if len(xs[mask]):
        dot_xy.set_data([xs[mask][-1]], [ys[mask][-1]])

    fig.suptitle(
        f"IMU Dead Reckoning  |  t = {t_now:.2f}s  |  |v| = {vel_mag[i]:.2f} m/s",
        color="white", fontsize=12
    )
    return line3d, dot3d, line_vm, vline_vm, line_xy, dot_xy


ani = animation.FuncAnimation(
    fig, update, frames=len(frames),
    init_func=init, interval=30, blit=True
)

plt.tight_layout()
plt.show()