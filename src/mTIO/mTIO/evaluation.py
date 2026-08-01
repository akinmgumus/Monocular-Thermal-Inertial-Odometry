"""
Trajectory evaluation utilities for the thermal-VIO pipeline.

Standard metrics for monocular VIO benchmarking:

  * ATE (Absolute Trajectory Error): per-sample position error after Sim(3)
    alignment of the VIO trajectory to ground truth (monocular has unobservable
    scale, so we use Umeyama with scale=True). Returns RMSE, mean, max, and the
    full time series for plotting.

  * RPE (Relative Pose Error): translation/rotation drift over a fixed time
    window (1 s by default). Less sensitive to global misalignment, captures
    local consistency. Typical values: < 0.1 m/s for indoor RGB VIO.

  * Drift rate: ATE_RMSE divided by trajectory length, expressed as a
    percentage. Standardised across automotive / aerial benchmarks. A 1-2 %
    drift rate is competitive for monocular VIO.

  * Survival time: how long before |position error| exceeds a threshold
    (default 2 m). For datasets where the pipeline diverges mid-run, this
    is the most honest single-number summary.

  * Filter consistency: cumulative used/in ratio of camera updates. Measures
    how often vision actually corrects the IMU integration. Below 0.3 the
    filter is effectively IMU-only.

A single results CSV accumulates one row per (dataset, method, config) run for
cross-dataset table generation. The 4-panel plot consolidates the whole story
into one figure: XY top-down, Z over time, ATE over time, track count over
time.
"""

import os
import csv
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


# ----------------------------------------------------------------------
# ALIGNMENT
# ----------------------------------------------------------------------
def umeyama_align(src: np.ndarray, dst: np.ndarray,
                  with_scale: bool = True, mode: str = None) -> dict:
    """Align `src` onto `dst` (both (N,3)) with the transform that matches the
    unobservable degrees of freedom of the sensor setup (Zhang & Scaramuzza 2018).

    mode:
      'sim3'   — 7 DOF (scale + SE(3)); monocular VO (scale unobservable).
      'se3'    — 6 DOF (rigid rotation + translation); stereo / RGB-D VO.
      'posyaw' — 4 DOF (yaw about gravity + translation); visual-INERTIAL
                 odometry. In VIO roll/pitch (via gravity) and scale (via the
                 accelerometer) ARE observable, so only global position and yaw
                 are free. A full SE(3) align would rotate the trajectory in
                 roll/pitch and thereby HIDE attitude drift the estimator is
                 meant to get right; posyaw leaves that error visible.
                 Assumes src and dst share a gravity-aligned (z-up) world frame.

    If mode is None it is derived from with_scale (back-compat): True→sim3,
    False→se3.

    Returns dict with R (3x3), s (float), t (3,), src_aligned (N,3), mode.
    Reference: Umeyama 1991; Zhang & Scaramuzza 2018 (alignment for VI-odometry).
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    if mode is None:
        mode = 'sim3' if with_scale else 'se3'

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    if mode == 'posyaw':
        # 4-DOF: best yaw about +z (gravity) for the xy-projection + translation.
        # psi = atan2( Σ(sx·dy − sy·dx), Σ(sx·dx + sy·dy) )
        sx, sy = src_c[:, 0], src_c[:, 1]
        dx, dy = dst_c[:, 0], dst_c[:, 1]
        psi = np.arctan2(np.sum(sx * dy - sy * dx), np.sum(sx * dx + sy * dy))
        c, sn = np.cos(psi), np.sin(psi)
        R_mat = np.array([[c, -sn, 0.0],
                          [sn,  c, 0.0],
                          [0.0, 0.0, 1.0]])
        s = 1.0
    else:
        cov = src_c.T @ dst_c / len(src)
        U, D, Vt = np.linalg.svd(cov)
        S = np.eye(3)
        if np.linalg.det(U @ Vt) < 0:
            S[2, 2] = -1.0
        R_mat = (Vt.T @ S @ U.T)
        if mode == 'sim3':
            var_src = (src_c ** 2).sum(axis=1).mean()
            s = (np.diag(D) @ S).trace() / var_src if var_src > 0 else 1.0
        else:  # 'se3'
            s = 1.0

    t = mu_dst - s * R_mat @ mu_src
    src_aligned = (s * R_mat @ src.T).T + t
    return {'R': R_mat, 's': float(s), 't': t,
            'src_aligned': src_aligned, 'mode': mode}


# ----------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------
def compute_ate(traj_aligned: np.ndarray, gt_at_traj: np.ndarray) -> dict:
    """Per-sample position error after alignment.

    traj_aligned : (N, 3) VIO positions after Sim(3) alignment to GT
    gt_at_traj   : (N, 3) GT positions interpolated to the same timestamps

    Returns dict with rmse, mean, median, max, and the (N,) error series.
    """
    err = np.linalg.norm(traj_aligned - gt_at_traj, axis=1)
    return {
        'rmse':   float(np.sqrt((err ** 2).mean())),
        'mean':   float(err.mean()),
        'median': float(np.median(err)),
        'max':    float(err.max()),
        'series': err,
    }


def compute_rpe(t: np.ndarray, traj: np.ndarray, traj_q: Optional[np.ndarray],
                gt_t: np.ndarray, gt_xyz: np.ndarray,
                gt_q: Optional[np.ndarray], window_s: float = 1.0) -> dict:
    """Relative pose error over a fixed-time window.

    For each pair (t, t+window_s), compare the VIO relative motion
    against the GT relative motion. Translation RPE = norm of difference.
    Rotation RPE = geodesic angle between relative rotations.

    traj_q / gt_q may be None to skip the rotational component.
    """
    out = {'trans_rmse': None, 'rot_rmse_deg': None,
           'trans_series': None, 'rot_series_deg': None}

    if len(t) < 3 or len(gt_t) < 3:
        return out

    pair_ends = []
    for i, ti in enumerate(t):
        j = int(np.searchsorted(t, ti + window_s))
        if j < len(t):
            pair_ends.append((i, j))
    if not pair_ends:
        return out

    vio_dpos = np.array([traj[j] - traj[i] for i, j in pair_ends])
    gt_dpos  = np.array([
        np.array([np.interp(t[j], gt_t, gt_xyz[:, k]) for k in range(3)])
        - np.array([np.interp(t[i], gt_t, gt_xyz[:, k]) for k in range(3)])
        for i, j in pair_ends
    ])
    trans_err = np.linalg.norm(vio_dpos - gt_dpos, axis=1)
    out['trans_rmse']  = float(np.sqrt((trans_err ** 2).mean()))
    out['trans_series'] = trans_err

    if traj_q is not None and gt_q is not None:
        rot_errs = []
        for i, j in pair_ends:
            R_vio_i = R.from_quat(traj_q[i]).as_matrix()
            R_vio_j = R.from_quat(traj_q[j]).as_matrix()
            R_vio_rel = R_vio_i.T @ R_vio_j
            q_gt_i = np.array([np.interp(t[i], gt_t, gt_q[:, k]) for k in range(4)])
            q_gt_j = np.array([np.interp(t[j], gt_t, gt_q[:, k]) for k in range(4)])
            q_gt_i /= np.linalg.norm(q_gt_i) + 1e-12
            q_gt_j /= np.linalg.norm(q_gt_j) + 1e-12
            R_gt_i = R.from_quat(q_gt_i).as_matrix()
            R_gt_j = R.from_quat(q_gt_j).as_matrix()
            R_gt_rel = R_gt_i.T @ R_gt_j
            R_diff = R_vio_rel.T @ R_gt_rel
            ang = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
            rot_errs.append(ang)
        rot_errs = np.array(rot_errs)
        out['rot_rmse_deg']  = float(np.sqrt((rot_errs ** 2).mean()))
        out['rot_series_deg'] = rot_errs
    return out


def compute_path_length(gt_xyz: np.ndarray) -> float:
    """Total path length of a trajectory (sum of consecutive segment norms)."""
    return float(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1).sum())


def compute_drift_rate(ate_rmse: float, path_length: float) -> float:
    """Drift rate as a percentage (ATE_RMSE / total_path_length × 100)."""
    return (ate_rmse / path_length * 100.0) if path_length > 0 else float('nan')


def compute_survival_time(t: np.ndarray, ate_series: np.ndarray,
                          threshold_m: float = 2.0) -> float:
    """First time at which |position error| exceeds the threshold. If never
    crossed, returns the full run duration. (Legacy: transient-sensitive; the
    pipeline now uses `detect_divergence` for the sustained-excursion rule.)"""
    over = np.where(ate_series > threshold_m)[0]
    if len(over) == 0:
        return float(t[-1] - t[0])
    return float(t[over[0]] - t[0])


def detect_divergence(t: np.ndarray, ate_series: np.ndarray,
                      threshold_m: float = 2.0,
                      min_duration_s: float = 1.0) -> dict:
    """First *sustained* divergence of the aligned trajectory.

    The divergence point is the earliest sample at which the 4-DOF-aligned
    position error exceeds `threshold_m` AND then stays above it for at least
    `min_duration_s` continuous seconds. A transient spike that recovers before
    `min_duration_s` has elapsed is ignored, so a single bad frame does not
    count as divergence.

    Both thresholds are parameters. `threshold_m` is an *absolute* position
    error, so it must be chosen relative to the scene scale (a 2 m threshold
    that is meaningful indoors is tiny for an 800 m outdoor flight).

    Returns dict:
        diverged : bool
        index    : int  — first sample index of the sustained excursion (or None)
        time     : float — absolute timestamp of that sample (or None)
    """
    over = np.asarray(ate_series) > threshold_m
    N = len(t)
    i = 0
    while i < N:
        if over[i]:
            j = i
            while j < N and over[j]:
                j += 1
            # contiguous excursion spans samples [i, j-1]
            if float(t[j - 1] - t[i]) >= min_duration_s:
                return {'diverged': True, 'index': int(i), 'time': float(t[i])}
            i = j
        else:
            i += 1
    return {'diverged': False, 'index': None, 'time': None}


# ----------------------------------------------------------------------
# ALL-IN-ONE WRAPPER
# ----------------------------------------------------------------------
def align_and_compute_metrics(
        traj: np.ndarray,                    # (N, 4): t, x, y, z
        gt_t: np.ndarray, gt_xyz: np.ndarray,
        with_scale: bool = False,
        align_mode: str = 'posyaw',
        settle_time_s: Optional[float] = None,
        divergence_time_s: Optional[float] = None,
) -> dict:
    """Align VIO to GT over the manually-marked good window, then compute metrics.

    Failure points are marked BY HAND (read off the X/Y/Z-vs-time plots), not
    auto-detected, because the transient behaviours differ per dataset and an
    absolute-error rule cannot tell them apart (an early low-excitation thrash
    and a large-but-constant offset both trip a threshold yet are not true
    divergence):
      * settle_time_s     — end of the initial start-up thrash; samples before
                            it are an expected low-excitation transient, not an
                            estimator fault, so they are excluded from the
                            tracking (pre-divergence) ATE. None -> start at t0.
      * divergence_time_s — where the estimate truly loses the trajectory
                            (unbounded growth). None -> never diverges.
    Both are RELATIVE seconds from the start, matching the origin-anchored time
    axis of the plots. The 4-DOF alignment is fit ONLY on the good window
    [settle, divergence], so neither the start-up thrash nor a post-divergence
    blow-up can corrupt it; the full-trajectory ATE is then measured against that
    fit (honest — the bad parts are allowed to show).

    Default alignment is 4-DOF 'posyaw' (yaw about gravity + translation), the
    correct convention for visual-inertial odometry (roll/pitch observable via
    gravity, scale via the accelerometer). Returns a flat dict for CSV writing.
    """
    t  = traj[:, 0]
    vp = traj[:, 1:4]
    t0 = float(t[0])
    N_pts = len(t)

    gI = np.stack([np.interp(t, gt_t, gt_xyz[:, k]) for k in range(3)], axis=1)

    def _apply(al, X):
        return (al['s'] * (al['R'] @ X.T)).T + al['t']

    # good tracking window [lo, hi) from the MANUAL marks only (relative
    # seconds). No auto-detection: a blank divergence mark means "never
    # diverged", so the window runs to the end. This is deliberate — the
    # transients (5-10 m start-up thrash before real tracking) make any fixed
    # threshold meaningless, so failure points are read off the plots by hand.
    lo = 0 if settle_time_s is None else int(np.searchsorted(t, t0 + settle_time_s))
    lo = max(0, min(lo, N_pts - 2))

    hi = N_pts if divergence_time_s is None else int(np.searchsorted(t, t0 + divergence_time_s))
    hi = max(lo + 2, min(hi, N_pts))

    align   = umeyama_align(vp[lo:hi], gI[lo:hi], with_scale=with_scale, mode=align_mode)
    aligned = _apply(align, vp)

    ate  = compute_ate(aligned, gI)                    # full trajectory
    good = compute_ate(aligned[lo:hi], gI[lo:hi])      # tracking window only
    rpe  = compute_rpe(t, aligned, None, gt_t, gt_xyz, None, window_s=1.0)
    path_len = compute_path_length(gI)
    drift = compute_drift_rate(ate['rmse'], path_len)

    diverged   = divergence_time_s is not None
    survival   = float(divergence_time_s) if diverged else float(t[-1] - t0)
    div_idx    = hi if (diverged and hi < N_pts) else None
    settle_idx = lo if (settle_time_s is not None and lo > 0) else None

    return {
        # Bookkeeping
        'n_samples':       len(t),
        'duration_s':      float(t[-1] - t0),
        'path_length_m':   path_len,
        'align_mode':      align['mode'],
        'align_scale':     align['s'],

        # ATE (full trajectory, against the good-window alignment)
        'ate_rmse_m':      ate['rmse'],
        'ate_mean_m':      ate['mean'],
        'ate_median_m':    ate['median'],
        'ate_max_m':       ate['max'],
        # tracking-window ATE = accuracy over [settle, divergence] only
        'pre_divergence_ate_rmse_m': good['rmse'],

        # RPE (internal fallback; the authoritative RPE is computed by evo)
        'rpe_trans_rmse':  rpe['trans_rmse'],
        'rpe_rot_rmse_deg': rpe['rot_rmse_deg'],

        # Derived
        'drift_pct':       drift,
        'survival_s':      survival,

        # Manual failure marks
        'diverged':          bool(diverged),
        'divergence_time_s': (float(divergence_time_s) if diverged else None),
        'settle_time_s':     (float(settle_time_s) if settle_idx is not None else None),

        # Series (not in CSV, used by plot). `aligned` is the FULL trajectory
        # under the good-window alignment.
        '_ate_series':     ate['series'],
        '_gt_at_traj':     gI,
        '_aligned_traj':   aligned,
        '_divergence_idx': div_idx,
        '_settle_idx':     settle_idx,
    }


# ----------------------------------------------------------------------
# FIGURE HELPERS
# ----------------------------------------------------------------------
def _alignment_label(metrics: dict) -> str:
    """Human-readable alignment label for figure titles."""
    mode = metrics.get('align_mode')
    if mode == 'posyaw':
        return 'posyaw (4-DOF)'
    if mode == 'se3':
        return 'SE(3)'
    if mode == 'sim3':
        return f"Sim(3), s={metrics.get('align_scale', 1.0):.3f}"
    # back-compat: infer from scale
    return ('SE(3)' if abs(metrics.get('align_scale', 1.0) - 1.0) < 1e-6
            else f"Sim(3), s={metrics['align_scale']:.3f}")


def _origin_anchored(metrics: dict, gt_xyz: np.ndarray):
    """Translate the aligned VIO and GT so that both start at the origin at
    t=0. Purely a plotting choice — the numbers in `metrics` were computed
    BEFORE this shift; it keeps the curves overlapping at the start so the
    reader sees shape and divergence without the alignment translation offset.

    Returns (gI_v, vp_a_v, gt_xyz_v): origin-shifted GT-at-traj, aligned VIO,
    and the full GT timeline.
    """
    gI   = metrics['_gt_at_traj']
    vp_a = metrics['_aligned_traj']
    gI_origin   = gI[0].copy()
    vp_a_origin = vp_a[0].copy()
    return gI - gI_origin, vp_a - vp_a_origin, gt_xyz - gI_origin


def _title_block(fig, title_info: dict, metrics: dict, subtitle: str) -> None:
    """Short config-id title only — NO metric values on the figure. The numbers
    live in results/all_runs.csv and the rendered summary table (render_summary_table).
    """
    cid = title_info.get('run_id')
    if not cid:
        ds    = title_info.get('dataset', '?')
        meth  = (title_info.get('method') or '?').upper()
        clahe = title_info.get('clahe')
        cl = ('CLAHE on'  if clahe in (True, 'on', 'On', 'ON') else
              'CLAHE off' if clahe is not None else '')
        cid = f"{ds} · {meth}" + (f" · {cl}" if cl else '')
    fig.suptitle(f"{cid}   [{subtitle}]", fontsize=12, y=0.995)


# ----------------------------------------------------------------------
# FIGURE 1 — TRAJECTORY: X(t), Y(t), Z(t) components + XY top-down
# ----------------------------------------------------------------------
def plot_trajectory(
        traj: np.ndarray,
        gt_t: np.ndarray, gt_xyz: np.ndarray,
        metrics: dict,
        title_info: dict,
        out_png: str,
        t_max: Optional[float] = None,
        gt_full_xyz: Optional[np.ndarray] = None,
) -> None:
    """Trajectory figure: the X, Y and Z position components over time against
    GT (left column, stacked, shared time axis), and the top-down XY trajectory
    against GT (right, square). If `t_max` (relative seconds) is given, the run
    is truncated there — used to draw the trajectory up to the divergence point
    so the tracked portion fills the frame instead of the post-divergence
    blow-up. All curves are $\\mathrm{SE}(3)$-aligned (for
    VIO) and origin-anchored at t=0.

    If `gt_full_xyz` (the whole-run GT positions, before any truncation) is
    given, the complete GT route is drawn faintly on the top-down XY panel and
    the estimate's final (divergence) point is starred, so the reader sees where
    on the true route the estimate broke off. Only meaningful together with
    `t_max` (a tracked-window figure).
    """
    if t_max is not None:                      # show only the tracked window
        # Slice to [settle, divergence] --- not [0, divergence] --- so the
        # start-up transient before settling is excluded, and re-anchor both
        # curves to the settle instant (below) so their start points coincide.
        t0 = float(traj[0, 0])
        t_min = metrics.get('settle_time_s') or 0.0
        lo = max(0, int(np.searchsorted(traj[:, 0], t0 + t_min)))
        hi = max(lo + 2, min(int(np.searchsorted(traj[:, 0], t0 + t_max)), len(traj)))
        traj = traj[lo:hi]
        gm = ((gt_t - t0) >= t_min) & ((gt_t - t0) <= t_max)
        gt_t, gt_xyz = gt_t[gm], gt_xyz[gm]
        metrics = dict(metrics)
        for _k in ('_ate_series', '_gt_at_traj', '_aligned_traj'):
            if metrics.get(_k) is not None:
                metrics[_k] = metrics[_k][lo:hi]
        # clean tracked view: the window boundaries are the frame edges, so the
        # settle/divergence markers and segment shading are dropped.
        metrics['_divergence_idx'] = None
        metrics['divergence_time_s'] = None
        metrics['_settle_idx'] = None
        metrics['settle_time_s'] = None
        metrics['_segments'] = None

    t      = traj[:, 0]
    t_rel  = t - t[0]
    gt_t_rel = gt_t - t[0]
    gI_v, vp_a_v, gt_xyz_v = _origin_anchored(metrics, gt_xyz)
    # whole-run GT route, anchored by the SAME origin as the windowed GT so it
    # overlays consistently on the top-down panel.
    gt_full_v = None
    if gt_full_xyz is not None:
        gI_origin = np.asarray(metrics['_gt_at_traj'])[0]
        gt_full_v = np.asarray(gt_full_xyz) - gI_origin
    align_label = _alignment_label(metrics)
    meth = title_info.get('method', '').upper()
    ds   = title_info.get('dataset', '')
    est_label = 'VIO' if 'euroc' in ds.lower() else 'TIO'

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.15])
    ax_x  = fig.add_subplot(gs[0, 0])
    ax_y  = fig.add_subplot(gs[1, 0], sharex=ax_x)
    ax_z  = fig.add_subplot(gs[2, 0], sharex=ax_x)
    ax_xy = fig.add_subplot(gs[:, 1])

    # manual failure marks (relative seconds; indices into traj / vp_a_v)
    div_idx    = metrics.get('_divergence_idx')
    settle_idx = metrics.get('_settle_idx')
    t_div      = metrics.get('divergence_time_s')
    t_settle   = metrics.get('settle_time_s')

    # -- X(t), Y(t), Z(t) components --
    # y-limits fit the UNION of GT and estimate so both curves stay fully on
    # screen; GT is drawn on top (higher zorder) so where they coincide it shows.
    for ax, k, name in ((ax_x, 0, 'x'), (ax_y, 1, 'y'), (ax_z, 2, 'z')):
        ax.plot(t_rel, vp_a_v[:, k], 'b-', lw=1.2, zorder=3,
                label=f'{est_label} ({meth})')
        ax.plot(gt_t_rel, gt_xyz_v[:, k], 'r--', lw=1.5, zorder=4, label='GT')
        allv = np.concatenate([vp_a_v[:, k], gt_xyz_v[:, k]])
        vmin, vmax = float(allv.min()), float(allv.max())
        span = max(vmax - vmin, 1e-3)
        ax.set_ylim(vmin - 0.15 * span, vmax + 0.15 * span)
        if t_settle is not None:
            ax.axvline(t_settle, color='g', ls=':', lw=1.0, alpha=0.7)
        if t_div is not None:
            ax.axvline(t_div, color='r', ls='--', lw=1.0, alpha=0.7)
        for sg in metrics.get('_segments') or []:      # segment shading + bounds
            if sg['segment_type'] == 'diverged':
                ax.axvspan(sg['t0'], sg['t1'], color='red', alpha=0.07, lw=0)
            if sg['segment_id'] > 0:
                ax.axvline(sg['t0'], color='gray', ls='-', lw=0.7, alpha=0.5)
        ax.set_ylabel(f'{name} [m]')
        ax.grid(alpha=0.3)
    ax_x.set_title('Position components over time (origin-anchored; y-limits fit GT and estimate)')
    ax_x.legend(loc='best', fontsize=9)
    ax_z.set_xlabel('t [s]')
    plt.setp(ax_x.get_xticklabels(), visible=False)
    plt.setp(ax_y.get_xticklabels(), visible=False)

    # -- XY top-down (square). Axes fit the UNION of GT and estimate so both are
    # fully visible; GT is drawn on top (higher zorder).
    if gt_full_v is not None:      # whole-run GT route for context (drawn behind)
        ax_xy.plot(gt_full_v[:, 0], gt_full_v[:, 1], '-', color='0.6', lw=1.3,
                   zorder=2, label='GT (full run)')
        ax_xy.scatter([gt_full_v[-1, 0]], [gt_full_v[-1, 1]], c='0.4', s=55,
                      marker='s', zorder=5, label='GT route end')
    ax_xy.plot(vp_a_v[:, 0], vp_a_v[:, 1], 'b-', lw=1.2, zorder=3,
               label=f'{est_label} ({meth})')
    ax_xy.plot(gI_v[:, 0], gI_v[:, 1], 'r--', lw=1.6, zorder=4,
               label='GT (tracked window)' if gt_full_v is not None else 'GT')
    ax_xy.scatter([0.0], [0.0], c='g', s=80, marker='o', zorder=5,
                  label='start (origin)')
    if gt_full_v is not None:
        # tracked figure with full-route context: the estimate's last sample is
        # the break-off (divergence) instant, and the windowed GT end is where GT
        # actually was at that moment.
        ax_xy.scatter([gI_v[-1, 0]], [gI_v[-1, 1]], c='k', s=80, marker='s',
                      zorder=5, label='GT at break-off')
        ax_xy.scatter([vp_a_v[-1, 0]], [vp_a_v[-1, 1]], c='r', s=150, marker='*',
                      zorder=6, edgecolors='k', linewidths=0.6,
                      label='divergence (break-off)')
    else:
        ax_xy.scatter([gI_v[-1, 0]], [gI_v[-1, 1]], c='k', s=70, marker='s',
                      zorder=5, label='GT end')
        ax_xy.scatter([vp_a_v[-1, 0]], [vp_a_v[-1, 1]], c='b', s=70, marker='X',
                      zorder=5, label=f'{est_label} end')
    if settle_idx is not None:
        ax_xy.scatter([vp_a_v[settle_idx, 0]], [vp_a_v[settle_idx, 1]], c='g',
                      s=110, marker='P', zorder=6, edgecolors='k', linewidths=0.6,
                      label='settle')
    if div_idx is not None:
        ax_xy.scatter([vp_a_v[div_idx, 0]], [vp_a_v[div_idx, 1]], c='r', s=120,
                      marker='*', zorder=6, edgecolors='k', linewidths=0.6,
                      label='divergence')
    ax_xy.set_xlabel('x [m]'); ax_xy.set_ylabel('y [m]')
    ax_xy.set_aspect('equal'); ax_xy.grid(alpha=0.3)
    # limits fit the union of GT and estimate (and the full route, when shown)
    # so nothing leaves the frame
    xs = [vp_a_v[:, 0], gI_v[:, 0]]
    ys = [vp_a_v[:, 1], gI_v[:, 1]]
    if gt_full_v is not None:
        xs.append(gt_full_v[:, 0]); ys.append(gt_full_v[:, 1])
    allx = np.concatenate(xs)
    ally = np.concatenate(ys)
    cx, cy = 0.5 * (allx.min() + allx.max()), 0.5 * (ally.min() + ally.max())
    half = 0.5 * max(allx.ptp(), ally.ptp(), 1e-3) * 1.12
    ax_xy.set_xlim(cx - half, cx + half)
    ax_xy.set_ylim(cy - half, cy + half * 1.15)
    ax_xy.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_xy.set_title(f'Top-down XY ({align_label}-aligned; limits fit GT and estimate)')

    _title_block(fig, title_info, metrics,
                 subtitle='trajectory (up to divergence)' if t_max is not None
                 else 'trajectory')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# FIGURE 2 — DIAGNOSTICS: ATE over time + front-end track counts over time
# ----------------------------------------------------------------------
def plot_diagnostics(
        traj: np.ndarray,
        metrics: dict,
        track_series: Optional[dict],     # {'t': [...], 'active': [...], 'used': [...]}
        title_info: dict,
        out_png: str,
) -> None:
    """Diagnostics figure: absolute trajectory error over time (top) and
    front-end track counts over time (bottom), sharing the time axis."""
    t     = traj[:, 0]
    t_rel = t - t[0]

    fig, (ax_ate, ax_tracks) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # -- ATE over time --
    ax_ate.plot(t_rel, metrics['_ate_series'], 'b-', lw=1.2)
    if metrics.get('seg_threshold_m') is not None:
        ax_ate.axhline(metrics['seg_threshold_m'], color='orange', ls='-.',
                       alpha=0.6, label=f"seg thr = {metrics['seg_threshold_m']:.1f} m")
    for sg in metrics.get('_segments') or []:          # segment shading + bounds
        if sg['segment_type'] == 'diverged':
            ax_ate.axvspan(sg['t0'], sg['t1'], color='red', alpha=0.07, lw=0)
        if sg['segment_id'] > 0:
            ax_ate.axvline(sg['t0'], color='gray', ls='-', lw=0.7, alpha=0.5)
    ax_ate.axhline(metrics['ate_rmse_m'], color='k', linestyle=':', alpha=0.5,
                   label='RMSE')
    if metrics.get('settle_time_s') is not None:
        ax_ate.axvline(metrics['settle_time_s'], color='g', linestyle=':',
                       alpha=0.7,
                       label='settle')
    if metrics.get('diverged'):
        ax_ate.axvline(metrics['divergence_time_s'], color='r', linestyle='--',
                       alpha=0.7,
                       label='divergence')
    ax_ate.set_ylabel('|err| [m]')
    ax_ate.grid(alpha=0.3); ax_ate.legend(loc='best', fontsize=9)
    ax_ate.set_title('Absolute trajectory error over time')

    # -- Front-end track counts over time --
    if track_series and len(track_series.get('t', [])) > 0:
        ts = np.asarray(track_series['t']) - t[0]
        if 'active' in track_series:
            ax_tracks.plot(ts, track_series['active'], 'g-', lw=1.0,
                           label=f"active tracks (μ={np.mean(track_series['active']):.0f})")
        if 'used' in track_series:
            ax_tracks.plot(ts, track_series['used'], 'b-', lw=1.0,
                           label=f"used in update (μ={np.mean(track_series['used']):.0f})")
        ax_tracks.set_ylabel('count'); ax_tracks.grid(alpha=0.3)
        ax_tracks.legend(loc='best', fontsize=9)
        ax_tracks.set_title('Front-end track counts over time')
    else:
        ax_tracks.text(0.5, 0.5, '(no track series captured)',
                       ha='center', va='center', transform=ax_tracks.transAxes,
                       fontsize=11, color='gray')
        ax_tracks.set_yticks([])
        ax_tracks.set_title('Front-end track counts over time')
    ax_tracks.set_xlabel('t [s]')

    _title_block(fig, title_info, metrics, subtitle='diagnostics')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# RESULTS CSV
# ----------------------------------------------------------------------
CSV_COLUMNS = [
    # identity + config
    'run_id',
    'dataset', 'front_end', 'clahe', 'norm_method', 'gauss_sigma',
    'chi2_confidence', 'N_max', 'mode',
    # accuracy
    'full_trajectory_ATE_RMSE', 'pre_divergence_ATE_RMSE', 'max_ATE',
    'drift_percent', 'RPE', 'path_length_m', 'survival_time_s',
    # manual failure marks
    'settle_time_s', 'diverged', 'divergence_time_s',
    # consistency
    'mean_nis', 'nis_pass_rate',
    # bookkeeping
    'n_samples', 'duration_s', 'align_mode',
    'avg_active_tracks', 'avg_used_per_update',
]


def build_csv_row(title_info: dict, metrics: dict) -> dict:
    """Map the run's `title_info` (config) and `metrics` (results) onto the
    reporting CSV schema (`CSV_COLUMNS`). Centralised so every dataset script
    writes an identical, comparable row. Missing keys become blank at write."""
    return {
        # identity + config
        'run_id':          title_info.get('run_id'),
        'dataset':         title_info.get('dataset'),
        'front_end':       (title_info.get('method') or '').upper(),
        'clahe':           title_info.get('clahe'),
        'norm_method':     title_info.get('normalize'),
        'gauss_sigma':     title_info.get('gaussian_sigma'),
        'chi2_confidence': title_info.get('chi2_alpha'),
        'N_max':           title_info.get('max_window'),
        'mode':            title_info.get('mode'),
        # accuracy. RPE prefers the evo value (set by regen); the internal
        # compute_rpe result is only a fallback for the first pass.
        'full_trajectory_ATE_RMSE': metrics.get('ate_rmse_m'),
        'pre_divergence_ATE_RMSE':  metrics.get('pre_divergence_ate_rmse_m'),
        'max_ATE':         metrics.get('ate_max_m'),
        'drift_percent':   metrics.get('drift_pct'),
        'RPE':             (metrics.get('rpe_evo')
                            if metrics.get('rpe_evo') is not None
                            else metrics.get('rpe_trans_rmse')),
        'path_length_m':   metrics.get('path_length_m'),
        'survival_time_s': metrics.get('survival_s'),
        # manual failure marks
        'settle_time_s':     metrics.get('settle_time_s'),
        'diverged':          metrics.get('diverged'),
        'divergence_time_s': metrics.get('divergence_time_s'),
        # consistency
        'mean_nis':        metrics.get('mean_nis'),
        'nis_pass_rate':   metrics.get('nis_pass_rate'),
        # bookkeeping
        'n_samples':       metrics.get('n_samples'),
        'duration_s':      metrics.get('duration_s'),
        'align_mode':      metrics.get('align_mode'),
        'avg_active_tracks':   metrics.get('avg_active_tracks'),
        'avg_used_per_update': metrics.get('avg_used_per_update'),
    }


def append_results_csv(csv_path: str, row: dict) -> None:
    """Upsert one row into the cumulative results CSV, keyed by `run_id`: a row
    with the same run_id is replaced (so a manual-mark regen overwrites the
    first-pass row rather than duplicating it). Missing columns are blanked."""
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)

    def _clean(r):
        c = {k: r.get(k, '') for k in CSV_COLUMNS}
        for k, v in list(c.items()):
            if isinstance(v, float):
                c[k] = round(v, 4)
        return c

    rid = row.get('run_id')
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            rows = [r for r in csv.DictReader(f)
                    if not (rid and r.get('run_id') == str(rid))]
    rows.append(_clean(row))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in CSV_COLUMNS})


# ----------------------------------------------------------------------
# MANUAL FAILURE MARKS  (results/divergence_marks.csv)
# ----------------------------------------------------------------------
# Hand-annotated per run, read off the X/Y/Z-vs-time plots. Columns:
#   run_id, settle_time_s, divergence_time_s      (empty cell -> None)
def load_divergence_mark(marks_csv: str, run_id: str):
    """Return (settle_time_s, divergence_time_s) for `run_id`, or (None, None)
    if the marks file or the row is absent. Empty cells map to None. Times are
    RELATIVE seconds, as read off the origin-anchored time plots."""
    if not os.path.exists(marks_csv):
        return None, None

    def _f(x):
        x = (x or '').strip()
        return float(x) if x else None

    with open(marks_csv, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('run_id') == run_id:
                return _f(r.get('settle_time_s')), _f(r.get('divergence_time_s'))
    return None, None


# ----------------------------------------------------------------------
# SUMMARY TABLE  (all runs -> one PNG)
# ----------------------------------------------------------------------
# (csv_column, header) pairs — the readable subset shown in the summary table.
SUMMARY_COLUMNS = [
    ('dataset',                  'dataset'),
    ('front_end',                'FE'),
    ('clahe',                    'CLAHE'),
    ('full_trajectory_ATE_RMSE', 'ATE [m]'),
    ('pre_divergence_ATE_RMSE',  'track ATE [m]'),
    ('RPE',                      'RPE [m/m]'),
    ('drift_percent',            'drift [%]'),
    ('survival_time_s',          'surv [s]'),
    ('diverged',                 'div'),
    ('divergence_time_s',        'div@ [s]'),
    ('mean_nis',                 'NIS/df'),
    ('nis_pass_rate',            'used/in'),
]


def _fmt_cell(x) -> str:
    """Compact cell text: blanks/None -> em dash, floats -> 4 significant figs."""
    s = str(x).strip()
    if s == '' or s.lower() == 'none':
        return '—'
    try:
        return f"{float(s):.4g}"
    except ValueError:
        return s


def render_summary_table(csv_path: str, out_png: str,
                         columns=None, title: str = None) -> None:
    """Render a results CSV as a single table PNG (thesis-ready), one row per
    run. Reads all rows, shows a readable subset of columns (SUMMARY_COLUMNS),
    sorted by dataset / front-end / CLAHE. The full data stays in the CSV."""
    columns = columns or SUMMARY_COLUMNS
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r.get('dataset', ''), r.get('front_end', ''),
                             str(r.get('clahe', ''))))

    headers = [h for _, h in columns]
    keys    = [k for k, _ in columns]
    cells   = [[_fmt_cell(r.get(k, '')) for k in keys] for r in rows] or [['—'] * len(keys)]

    fig_h = max(1.4, 0.34 * len(rows) + 0.9)
    fig, ax = plt.subplots(figsize=(1.05 * len(headers) + 1.5, fig_h))
    ax.axis('off')
    tbl = ax.table(cellText=cells, colLabels=headers, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    for j in range(len(headers)):                       # header row styling
        c = tbl[0, j]
        c.set_facecolor('#2c3e50')
        c.set_text_props(color='white', weight='bold')
    for i in range(1, len(cells) + 1):                  # zebra striping
        if i % 2 == 0:
            for j in range(len(headers)):
                tbl[i, j].set_facecolor('#f2f4f6')
    ax.set_title(title or 'MSCKF Thermal-Inertial Odometry — run summary',
                 fontsize=11, pad=12)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# SEGMENT-WISE ANALYSIS  (fixed per-dataset threshold, no auto-tuning)
# ----------------------------------------------------------------------
def segment_error_series(t, err, threshold_m: float, min_duration_s: float):
    """Split the aligned |err|(t) series into alternating 'tracking' /
    'diverged' segments with a FIXED threshold. An excursion above the
    threshold shorter than min_duration_s is a transient (stays tracking);
    symmetrically, a dip below shorter than min_duration_s between two
    above-regions does not re-enter tracking. Returns a list of dicts with
    segment_id, segment_type, i0, i1 (exclusive), t0, t1 (relative seconds)."""
    err = np.asarray(err)
    N = len(t)
    over = err > threshold_m
    # runs of constant state
    runs = []
    i = 0
    while i < N:
        j = i
        while j < N and over[j] == over[i]:
            j += 1
        runs.append([i, j, bool(over[i])])
        i = j
    # pass 1: short over-runs -> tracking (transient); pass 2: short under-runs
    # sandwiched between over-runs -> diverged (no real re-entry)
    for p in (True, False):
        merged = []
        for k, (s, e, st) in enumerate(runs):
            dur = float(t[e - 1] - t[s])
            if dur < min_duration_s:
                if p and st:
                    st = False
                elif not p and not st and 0 < k < len(runs) - 1:
                    st = True
            if merged and merged[-1][2] == st:
                merged[-1][1] = e
            else:
                merged.append([s, e, st])
        runs = merged
    t0 = float(t[0])
    return [{'segment_id': k,
             'segment_type': 'diverged' if st else 'tracking',
             'i0': int(s), 'i1': int(e),
             't0': float(t[s] - t0), 't1': float(t[e - 1] - t0)}
            for k, (s, e, st) in enumerate(runs)]


SEGMENTS_CSV_COLUMNS = [
    'run_id', 'segment_id', 'segment_type', 't_start_s', 't_end_s',
    'duration_s', 'n_samples', 'ate_rmse_m', 'rpe_trans_rmse', 'rpe_evo',
    'mean_nis', 'avg_active_tracks', 'avg_used_per_update',
]


def append_segments_csv(csv_path: str, run_id: str, seg_rows: list) -> None:
    """Upsert this run's segment rows (old rows of the same run_id dropped)."""
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            rows = [r for r in csv.DictReader(f) if r.get('run_id') != run_id]
    rows += [{k: ('' if r.get(k) is None else r.get(k)) for k in SEGMENTS_CSV_COLUMNS}
             for r in seg_rows]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SEGMENTS_CSV_COLUMNS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            out = {}
            for k in SEGMENTS_CSV_COLUMNS:
                v = r.get(k, '')
                out[k] = round(v, 4) if isinstance(v, float) else v
            w.writerow(out)


def run_segment_analysis(run_id, traj, metrics, gt_t, gt_xyz,
                         threshold_m, min_duration_s,
                         track_series=None, nis_log=None,
                         csv_path='results/segments.csv'):
    """Segment the run and compute per-segment metrics.

    ATE uses the samples of the segment only, under the run's single alignment
    (metrics['_aligned_traj']); RPE here is the internal fallback (evo fills
    'rpe_evo' at regen); mean NIS/df uses only tracks accepted inside the
    segment's time span; track counts are averaged over the span. Rows are
    upserted into `csv_path`, segments stored in metrics['_segments'] so the
    plots can shade them. Returns the segment rows."""
    t   = traj[:, 0]
    err = metrics['_ate_series']
    aligned = metrics['_aligned_traj']
    segs = segment_error_series(t, err, threshold_m, min_duration_s)
    metrics['_segments'] = segs
    metrics['seg_threshold_m'] = float(threshold_m)

    nis = ([(tt, d2, df) for tt, d2, df in (nis_log or []) if tt is not None])
    ts_t = np.asarray(track_series['t']) if track_series else None

    rows = []
    for sg in segs:
        s, e = sg['i0'], sg['i1']
        row = {'run_id': run_id, 'segment_id': sg['segment_id'],
               'segment_type': sg['segment_type'],
               't_start_s': sg['t0'], 't_end_s': sg['t1'],
               'duration_s': sg['t1'] - sg['t0'], 'n_samples': e - s,
               'ate_rmse_m': float(np.sqrt((err[s:e] ** 2).mean()))}
        rpe = compute_rpe(t[s:e], aligned[s:e], None, gt_t, gt_xyz, None,
                          window_s=1.0)
        row['rpe_trans_rmse'] = rpe['trans_rmse']
        lo_t, hi_t = float(t[s]), float(t[e - 1])
        seg_nis = [(d2, df) for tt, d2, df in nis if lo_t <= tt <= hi_t]
        sdf = sum(df for _, df in seg_nis)
        row['mean_nis'] = (sum(d2 for d2, _ in seg_nis) / sdf) if sdf else None
        if ts_t is not None and len(ts_t):
            m = (ts_t >= lo_t) & (ts_t <= hi_t)
            if m.any():
                row['avg_active_tracks']   = float(np.asarray(track_series['active'])[m].mean())
                row['avg_used_per_update'] = float(np.asarray(track_series['used'])[m].mean())
        rows.append(row)

    append_segments_csv(csv_path, run_id, rows)
    print(f"\nsegments (thr={threshold_m} m, min_dur={min_duration_s} s):")
    for r in rows:
        mn = f"{r['mean_nis']:.2f}" if r.get('mean_nis') else "—"
        print(f"  #{r['segment_id']} {r['segment_type']:9s} "
              f"[{r['t_start_s']:7.1f},{r['t_end_s']:7.1f}]s  "
              f"ATE={r['ate_rmse_m']:.3f}m  NIS/df={mn}")
    return rows
