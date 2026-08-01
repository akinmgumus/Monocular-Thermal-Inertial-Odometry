"""Compute ATE/RPE with evo, compare against our internal metrics, and
collect the results into a CSV.

Evaluate a single run:
    python3 scripts/evo_eval.py results/msckf_vio_sthereo_valley_klt_clahe_on.txt sthereo
      -> prints evo + our own ATE/RPE AND appends a row to
         results/evo_runs.csv (upserts by 'run' name, keeping the latest).

Table (from all collected runs):
    python3 scripts/evo_eval.py table
      -> prints results/evo_runs.csv as a markdown + LaTeX booktabs table.

Notes:
  - If the saved trajectory has 8 columns (t x y z qx qy qz qw), RPE uses the
    real orientation; with 4 columns the estimate gets an identity
    quaternion -> ATE is valid, RPE is not.
  - PoseTrajectory3D expects quaternion order w,x,y,z; scipy/TUM use x,y,z,w.
  - SE(3) alignment: align(ref, correct_scale=False).
"""
import os
import sys
import csv
import importlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/mTIO'))

from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from mTIO.evaluation import (
    align_and_compute_metrics, load_divergence_mark, build_csv_row,
    append_results_csv, plot_trajectory, plot_diagnostics, render_summary_table,
    run_segment_analysis, append_segments_csv,
)

MARKS_CSV    = 'results/divergence_marks.csv'   # hand-annotated settle/divergence
ALL_RUNS_CSV = 'results/all_runs.csv'           # full-schema results (upsert by run_id)

CONFIGS = {
    'rovtio':     'config_rovtio',
    'firestereo': 'config_firestereo',
    'sthereo':    'config_sthereo',
    'euroc':      'config_euroc',
    'voxlbag':    'config_voxlbag',
}

CSV_PATH = 'results/evo_runs.csv'
CSV_COLUMNS = [
    'dataset', 'run', 'method', 'clahe', 'preproc', 'n_pairs',
    'ate_rmse_posyaw',         # ATE — 4-DOF posyaw (VIO-correct, headline; ours)
    'rpe_rmse_evo_per_m',      # RPE — evo (standard, alignment-invariant)
]


def _xyzw_to_wxyz(q):
    q = np.asarray(q, float)
    return q[:, [3, 0, 1, 2]]


def load_traj(path):
    """Saved traj -> (t, xyz, quat_xyzw, has_q). Identity quaternion if 4 columns."""
    d = np.loadtxt(path)
    t, xyz = d[:, 0], d[:, 1:4]
    if d.shape[1] >= 8:
        return t, xyz, d[:, 4:8], True
    return t, xyz, np.tile([0, 0, 0, 1.0], (len(d), 1)), False


def parse_run_meta(path):
    """Infer method / clahe / preproc from the filename (best-effort)."""
    name = os.path.splitext(os.path.basename(path))[0]
    low  = name.lower()
    method = 'orb' if 'orb' in low else 'klt' if 'klt' in low else ''
    if 'clahe_on' in low or 'claheon' in low:
        clahe = 'on'
    elif 'clahe_off' in low or 'claheoff' in low:
        clahe = 'off'
    else:
        clahe = ''
    if 'normclahe' in low:
        preproc = 'normClahe'      # dataloader2 (norm→CLAHE)
    elif 'clahenorm' in low:
        preproc = 'claheNorm'      # dataloader  (CLAHE→norm)
    else:
        preproc = 'default'
    return name, method, clahe, preproc


def append_csv(row):
    """Append row to the CSV; upsert by 'run' name, keeping the latest."""
    os.makedirs(os.path.dirname(CSV_PATH) or '.', exist_ok=True)
    rows = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline='') as f:
            rows = [r for r in csv.DictReader(f) if r.get('run') != row['run']]
    rows.append({k: row.get(k, '') for k in CSV_COLUMNS})
    with open(CSV_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in CSV_COLUMNS})


def _read_row(csv_path, run_id):
    """Existing full-schema row for run_id (dict of strings), or {} if absent.
    Used to preserve fields regen cannot recompute from a saved trajectory
    (mean_nis, nis_pass_rate, track counts, and the config columns)."""
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('run_id') == run_id:
                return r
    return {}


def evaluate(traj_path, ds):
    config = importlib.import_module(f"mTIO.{CONFIGS[ds]}")
    t, xyz, quat, has_q = load_traj(traj_path)
    gt = config.load_ground_truth()

    # GT quaternion order differs by dataset: EuRoC stores wxyz, all others
    # xyzw. Passing the wrong order silently corrupts the RPE (which uses
    # orientation) while leaving ATE untouched, so convert per dataset.
    gt_q_wxyz = gt['q'] if ds == 'euroc' else _xyzw_to_wxyz(gt['q'])
    est = PoseTrajectory3D(positions_xyz=xyz,
                           orientations_quat_wxyz=_xyzw_to_wxyz(quat), timestamps=t)
    ref = PoseTrajectory3D(positions_xyz=gt['p'],
                           orientations_quat_wxyz=gt_q_wxyz, timestamps=gt['t'])
    ref_s, est_s = sync.associate_trajectories(ref, est, max_diff=0.05)
    print(f"associated pairs: {ref_s.num_poses}  (est {est.num_poses}, gt {ref.num_poses})")
    est_s.align(ref_s, correct_scale=False)

    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref_s, est_s))
    S = metrics.StatisticsType
    ate = {k: ape.get_statistic(getattr(S, k)) for k in ('rmse', 'mean', 'median', 'max')}

    rpe = metrics.RPE(metrics.PoseRelation.translation_part,
                      delta=1.0, delta_unit=metrics.Unit.meters, all_pairs=False)
    rpe.process_data((ref_s, est_s))
    rpe_rmse = rpe.get_statistic(S.rmse)

    name, method, clahe, preproc = parse_run_meta(traj_path)

    # manual failure marks (hand-annotated, read off the X/Y/Z-vs-time plots)
    settle, diverge = load_divergence_mark(MARKS_CSV, name)

    traj4 = np.column_stack([t, xyz])
    ours = align_and_compute_metrics(traj4, gt['t'], gt['p'],
                                     settle_time_s=settle, divergence_time_s=diverge)
    ours_se3 = align_and_compute_metrics(traj4, gt['t'], gt['p'], align_mode='se3')
    ours['rpe_evo'] = float(rpe_rmse)      # authoritative RPE (evo, per-metre)

    # segment analysis (fixed per-dataset thresholds) + per-segment evo RPE:
    # slice the estimate by each segment's time span, rerun evo RPE on it.
    # NIS / track counts only exist in the live run's rows — snapshot them
    # BEFORE run_segment_analysis upserts, and merge back by segment_id.
    old_segs = {}
    if os.path.exists('results/segments.csv'):
        with open('results/segments.csv', newline='') as fh:
            for r in csv.DictReader(fh):
                if r.get('run_id') == name:
                    old_segs[r['segment_id']] = r
    # track counts saved by the live run (sidecar) → restore for diag + segments
    trk = None
    tp = f"results/{name}_tracks.npz"
    if os.path.exists(tp):
        z = np.load(tp); trk = {k: z[k] for k in z.files}
    seg_rows = run_segment_analysis(
        name, traj4, ours, gt['t'], gt['p'],
        config.SEG_THRESHOLD_M, config.SEG_MIN_DURATION_S,
        track_series=trk, nis_log=None)
    t0 = t[0]
    for sr, sg in zip(seg_rows, ours['_segments']):
        m = (t >= t0 + sg['t0']) & (t <= t0 + sg['t1'])
        if m.sum() < 10:
            continue
        est_seg = PoseTrajectory3D(positions_xyz=xyz[m],
                                   orientations_quat_wxyz=_xyzw_to_wxyz(quat[m]),
                                   timestamps=t[m])
        try:
            ref_g, est_g = sync.associate_trajectories(ref, est_seg, max_diff=0.05)
            est_g.align(ref_g, correct_scale=False)
            r = metrics.RPE(metrics.PoseRelation.translation_part, delta=1.0,
                            delta_unit=metrics.Unit.meters, all_pairs=False)
            r.process_data((ref_g, est_g))
            sr['rpe_evo'] = float(r.get_statistic(S.rmse))
        except Exception as ex:
            print(f"  [seg {sg['segment_id']}] evo RPE atlandi: {ex}")
    for sr in seg_rows:
        old = old_segs.get(str(sr['segment_id']), {})
        for k in ('mean_nis', 'avg_active_tracks', 'avg_used_per_update'):
            if sr.get(k) in (None, '') and old.get(k) not in (None, ''):
                sr[k] = old[k]
    append_segments_csv('results/segments.csv', name, seg_rows)

    print("\n=== REPORTED METRICS ===")
    print(f"  marks: settle={settle}  divergence={diverge}")
    print(f"  ATE rmse  [m]   (posyaw, 4-DOF, ours)  : {ours['ate_rmse_m']:.3f}")
    print(f"  track ATE [m]   (settle..divergence)   : {ours['pre_divergence_ate_rmse_m']:.3f}")
    print(f"  RPE rmse  [m/m] (evo, standard)        : {rpe_rmse:.3f}")
    # SE(3) is NOT USED in the report — only to validate our own ATE
    # machinery against evo during development (ours SE(3) ~ evo SE(3) should hold).
    print(f"  [validation] SE(3): ours {ours_se3['ate_rmse_m']:.3f}  vs  evo {ate['rmse']:.3f}"
          f"  (not part of the report)")
    if not has_q:
        print("\n[!] no est quaternion (identity) -> ATE is reliable, RPE is not (save 8 columns).")

    # regenerate the marked figures and upsert the full-schema results row.
    title_info = {'run_id': name, 'dataset': ds, 'method': method,
                  'clahe': clahe, 'normalize': preproc, 'mode': 'vio'}
    base = os.path.join('results', name)
    plot_trajectory(traj4, gt['t'], gt['p'], ours, title_info, base + '_traj.png')
    if diverge is not None:      # extra figure: trajectory only up to divergence
        plot_trajectory(traj4, gt['t'], gt['p'], ours, title_info,
                        base + '_traj_tracked.png', t_max=diverge)
    plot_diagnostics(traj4, ours, trk, title_info, base + '_diag.png')

    # Preserve fields regen cannot recompute from a saved trajectory (filter
    # consistency + the config columns): start from the first-pass row and only
    # overwrite what regen produces.
    row  = build_csv_row(title_info, ours)
    prev = _read_row(ALL_RUNS_CSV, name)
    for k, v in prev.items():
        if (row.get(k) in (None, '')) and v not in (None, ''):
            row[k] = v
    append_results_csv(ALL_RUNS_CSV, row)
    print(f"regenerated {base}_traj.png / _diag.png  +  upserted → {ALL_RUNS_CSV}")

    # keep the lightweight evo_runs.csv too (headline ATE + RPE)
    append_csv({
        'dataset': ds, 'run': name, 'method': method, 'clahe': clahe,
        'preproc': preproc, 'n_pairs': ref_s.num_poses,
        'ate_rmse_posyaw':    f"{ours['ate_rmse_m']:.4f}",
        'rpe_rmse_evo_per_m': f"{rpe_rmse:.4f}",
    })
    print(f"appended → {CSV_PATH}")


def print_table():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"{CSV_PATH} does not exist — evaluate at least one run first.")
    with open(CSV_PATH, newline='') as f:
        rows = sorted(csv.DictReader(f),
                      key=lambda r: (r['dataset'], r['method'], r['clahe'], r['preproc']))

    print("\n--- markdown (ATE = posyaw 4-DOF; RPE = evo) ---")
    print("| dataset | method | clahe | preproc | ATE [m] | RPE [m/m] |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['method']} | {r['clahe']} | {r['preproc']} "
              f"| {float(r['ate_rmse_posyaw']):.2f} | {float(r['rpe_rmse_evo_per_m']):.3f} |")

    print("\n--- LaTeX (booktabs) ---")
    print(r"\begin{tabular}{llllrr}")
    print(r"\toprule")
    print(r"Dataset & Method & CLAHE & Preproc & ATE [m] & RPE [m/m] \\")
    print(r"\midrule")
    for r in rows:
        print(f"{r['dataset']} & {r['method']} & {r['clahe']} & {r['preproc']} & "
              f"{float(r['ate_rmse_posyaw']):.2f} & {float(r['rpe_rmse_evo_per_m']):.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == 'table':
        print_table()
        return
    traj_path = sys.argv[1]
    ds = sys.argv[2] if len(sys.argv) > 2 else 'rovtio'
    evaluate(traj_path, ds)


if __name__ == '__main__':
    main()
