# Monocular Thermal-Inertial Odometry (TIO)

Monocular MSCKF-based thermal-inertial odometry, evaluated across an RGB sanity-check baseline and three thermal datasets spanning indoor/outdoor flight and ground-vehicle driving.

This is the codebase for an M.Sc. thesis. The contribution is not a new algorithm but a reproducible thermal-TIO evaluation harness that decouples three layers — preprocessing, front-end (KLT / ORB), and back-end (MSCKF) — and runs the same filter against multiple datasets with a single shared parameter set, so that differences in tracking performance are attributable to the data rather than to per-dataset tuning.

## Quick start

```bash
# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e src/thermal_vo

# Choose a dataset, extract its bag to flat layout, then run the test script.
# Example for ROVTIO:
python3 scripts/extract_rovtio_bag.py \
    --bag-dir ~/Downloads/alt1 \
    --out ~/rovtio_data/processed/alt1

python3 scripts/test_msckf_tio_rovtio.py
```

Each test script propagates the MSCKF over the whole run and, at the end, computes ATE / RPE / survival-time / filter-consistency metrics, saves a trajectory figure and a diagnostics figure, and appends one row to `results/all_runs.csv` for cross-dataset comparison.

### Configuration (environment variables)

Every test script shares the same knobs, read at import time:

| Variable | Values | Meaning |
|---|---|---|
| `TVIO_METHOD` | `orb` (default) / `klt` | feature front-end |
| `TVIO_USE_CLAHE` | `0` / `1` | contrast enhancement on/off |
| `TVIO_DL_VER` | `1` / `2` | preprocessing order: `1` = CLAHE→normalise, `2` = normalise→CLAHE (adopted) |
| `TVIO_LIVE_PLOT` | `0` (default) / `1` | live-updating matplotlib view vs. silent run + saved PNG |

```bash
TVIO_METHOD=klt TVIO_USE_CLAHE=1 TVIO_DL_VER=2 python3 scripts/test_msckf_tio_sthereo.py
```

`scripts/run_all_tests.py` sweeps a fixed set of (method, CLAHE, ordering) permutations across every dataset in one go.

## Repository layout

```
tvio_ws/
├── scripts/
│   ├── extract_firestereo_bag.py      # ROS1 .bag → flat layout (PNG + IMU CSV + GT)
│   ├── extract_rovtio_bag.py
│   ├── extract_voxlbag.py
│   ├── test_msckf_vio_euroc.py        # RGB visual-inertial baseline (mocap GT) — sanity check, not thermal
│   ├── test_msckf_tio_firestereo.py   # Outdoor monocular thermal (LiDAR-inertial pseudo-GT)
│   ├── test_msckf_tio_rovtio.py       # Indoor thermal (Vicon GT)
│   ├── test_msckf_tio_sthereo.py      # Vehicle thermal (local-pose reference)
│   ├── test_msckf_tio_voxlbag.py      # Outdoor drone thermal (PX4 EKF2 reference; excluded from
│   │                                   #   the thesis's core results — unresolved IMU axis convention)
│   ├── run_all_tests.py               # sweeps (front-end × CLAHE × ordering) across datasets
│   ├── evo_eval.py                    # regen: recompute ATE/RPE (evo) + figures from a saved
│   │                                   #   trajectory .txt, without re-running the filter
│   ├── build_result_tables.py         # recomputes metrics for a fixed run list and emits LaTeX tables
│   └── fig_*.py                       # figure-generation scripts for specific analyses
│
├── src/thermal_vo/
│   ├── setup.py
│   └── thermal_vo/
│       ├── config_euroc.py            # per-dataset calibration + IMU noise + GT loader
│       ├── config_firestereo.py
│       ├── config_rovtio.py
│       ├── config_sthereo.py
│       ├── config_voxlbag.py
│       ├── common_params.py           # shared front-end/back-end parameters (single source for all datasets)
│       ├── dataloader.py              # ThermalDataLoader + IMULoader + VIOSequencer (CLAHE → normalise)
│       ├── dataloader2.py             # same, with normalise → CLAHE (the adopted ordering)
│       ├── klt.py                     # KLT optical-flow tracker (thermal-tuned)
│       ├── orb.py                     # ORB descriptor tracker, grid-distributed (thermal-tuned)
│       ├── msckf.py                   # MSCKF filter: predict / augment / sequential update / prune, with FEJ
│       └── evaluation.py              # ATE (posyaw 4-DOF) / RPE (evo) / survival / NIS, plots, CSV
│
├── results/                           # generated (gitignored): trajectories, plots, CSV, divergence marks
├── data/                              # raw dataset bags (gitignored)
├── requirements.txt
└── .gitignore
```

## Pipeline overview

```
Thermal PNG ─┐                          ┌─ KLT optical flow ─┐
             ├─ ThermalDataLoader        │                    │
IMU CSV   ─┐ │  (undistort + denoise +  ─┤                    │
           └─┤   normalise + CLAHE)      └─ ORB descriptor ───┤
GT        ───┤                                                │
             │           active / dead feature tracks          │
             │                                                ▼
             │                                    ┌── MSCKF back-end ──┐
             │                                    │ predict  (IMU, FEJ)│
             ▼                                    │ augment_state      │
  VIOSequencer (chronological IMU + cam events) ─►│ triangulate (GN)   │
                                                  │ parallax + χ² gates│
                                                  │ sequential EKF upd.│
                                                  │ prune cam_states   │
                                                  └────────┬───────────┘
                                                           ▼
                                              {pose, velocity, biases}
                                                           │
                                                           ▼
                                          thermal_vo.evaluation
                                          (posyaw 4-DOF ATE, evo RPE,
                                           survival time, NIS/df,
                                           trajectory + diagnostics
                                           figures, append to CSV)
```

Two back-end design choices worth calling out:
- **FEJ (First-Estimate Jacobians)**: every Jacobian is linearised at each state's frozen first estimate while the residual uses the current estimate, which keeps the unobservable directions (global position and yaw) consistent and stops a spurious yaw-drift feedback loop that a naive EKF-MSCKF exhibits.
- **Sequential per-track update**: instead of stacking every accepted track into one giant measurement and applying a single Kalman gain, each track updates the covariance individually. Batching hundreds of tracks in one step was found to shrink the covariance so aggressively that the next update's χ² gate rejected everything (a "chi-square cascade"), collapsing the filter to pure IMU dead-reckoning.

## Metrics (reported per run)

| Metric | Definition |
|---|---|
| ATE (full / tracking-window) | Absolute trajectory error, RMS of position residuals after 4-DOF (position + yaw) alignment — the correct convention for VIO/TIO, since roll/pitch are observable via gravity and scale via the accelerometer. Reported both over the whole run and over the manually identified tracking window `[settle, divergence]` |
| RPE | Relative pose error (translation only), computed with `evo` over consecutive 1-metre sub-segments of travelled distance — a distance-normalised, alignment-invariant metric, reported in m/m |
| Survival time | Time from run start to the divergence instant (manually identified from the trajectory/error plots; see `results/divergence_marks.csv`) |
| NIS/df | Time-averaged normalised innovation squared per degree of freedom, over χ²-gate-accepted tracks — ≈1 indicates a consistent filter |
| used/in | Fraction of candidate tracks accepted by the χ² gate — a coarser filter-consistency proxy |
| avg active / avg used tracks | Front-end health: tracks currently followed vs. tracks actually incorporated per update |

All metrics are written to `results/all_runs.csv` for cross-dataset table generation (`build_result_tables.py`).

## Datasets

| Dataset | Modality | GT type | Platform | Where to obtain |
|---|---|---|---|---|
| EuRoC MH_03_medium | 8-bit grayscale (visible) | Vicon mocap | UAV indoor | https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets |
| FIReStereo frick_1 | 16-bit thermal (Boson+) | LiDAR-inertial (pseudo-GT) | UAV outdoor wildfire | https://github.com/CMU-Wilderness/FireStereo |
| ROVTIO alt1 | 16-bit thermal (Tau2) | Vicon mocap | UAV indoor | https://huggingface.co/datasets/ntnu-arl/rovtio |
| SThereo valley_evening | 14-bit thermal | Local-pose (pseudo-GT) | Ground vehicle | https://sites.google.com/view/sthereo |
| VOXL Starling 2 Max (voxlbag) | 8-bit thermal (Boson+, H.264-decoded) | PX4 EKF2 / GPS | UAV outdoor | private bag, not publicly released |

EuRoC is a visible-light, non-thermal baseline: it validates that the MSCKF back-end itself is correct (RQ1), so that any degradation seen on the thermal datasets is attributable to the thermal modality rather than to an implementation fault. The VOXL bag is integrated in the harness but excluded from the thesis's core cross-dataset results, owing to an unresolved IMU axis-convention discrepancy.

## License

See `LICENSE`.
