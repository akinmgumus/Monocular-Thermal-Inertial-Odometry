# Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e src/mTIO

# Extract a dataset bag to the flat layout, then run its test script.
python3 scripts/extract_rovtio_bag.py --bag-dir ~/Downloads/alt1 --out ~/rovtio_data/processed/alt1
python3 scripts/test_msckf_tio_rovtio.py
```

Each test script runs the filter over the whole sequence, computes the metrics, saves a trajectory figure and a diagnostics figure, and adds one row to `results/all_runs.csv`.

## Settings

All the test scripts read the same environment variables at import time.

| Variable | Values | Meaning |
|---|---|---|
| `TVIO_METHOD` | `orb` (default) / `klt` | feature front-end |
| `TVIO_USE_CLAHE` | `0` / `1` | contrast enhancement on/off |
| `TVIO_DL_VER` | `1` / `2` | preprocessing order: `1` = CLAHE→normalise, `2` = normalise→CLAHE (the one used) |
| `TVIO_LIVE_PLOT` | `0` (default) / `1` | live matplotlib window vs. silent run with a saved PNG |

```bash
TVIO_METHOD=klt TVIO_USE_CLAHE=1 TVIO_DL_VER=2 python3 scripts/test_msckf_tio_sthereo.py
```

`scripts/run_all_tests.py` sweeps front-end × CLAHE × ordering across every dataset in one go.

## Layout

```
scripts/          bag extractors (ROS1 .bag → flat PNG + IMU CSV + GT), one test script
                  per dataset, the sweep runner, an evo re-evaluation entry point and
                  figure-generation scripts
src/mTIO/         config_<dataset>.py, common_params.py  calibration, IMU noise, GT
                  dataloader.py / dataloader2.py         preprocessing + VIOSequencer
                  klt.py / orb.py                        thermal-tuned front-ends
                  msckf.py                               the filter (FEJ, sequential upd.)
                  evaluation.py                          posyaw ATE, evo RPE, plots, CSV
docs/figures/     figures used in the README
results/, data/   generated outputs and raw bags — both gitignored
```

## Metrics recorded per run

Beyond ATE and RPE, each run stores **survival time** (start to divergence), **NIS/df** (time-averaged normalised innovation squared per degree of freedom over χ²-accepted tracks; around 1 means a consistent filter), and **average active / used track counts** as a front-end health check. Everything lands in `results/all_runs.csv`.

## Datasets

| Dataset | Modality | Ground truth | Platform | Source |
|---|---|---|---|---|
| EuRoC MH_03_medium | 8-bit grayscale (visible) | Vicon mocap | UAV indoor | [ETH ASL](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) |
| FIReStereo frick_1 | 16-bit thermal (Boson+) | LiDAR-inertial | UAV outdoor | [GitHub](https://github.com/CMU-Wilderness/FireStereo) |
| ROVTIO alt1 | 16-bit thermal (Tau2) | Vicon mocap | UAV indoor | [HuggingFace](https://huggingface.co/datasets/ntnu-arl/rovtio) |
| SThereo valley_evening | 14-bit thermal | Local-pose | Ground vehicle | [Project site](https://sites.google.com/view/sthereo) |
| AerialTN | 8-bit thermal (Boson+) | PX4 EKF2 / GPS | UAV outdoor | private bag, not released |

AerialTN files are named `*_voxlbag.*` in the code, after the VOXL bag format the sequence was recorded in.
