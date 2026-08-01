# Monocular Thermal-Inertial Odometry (mTIO)

Where does a thermal camera actually work for state estimation, and where does it fall over? This is an MSCKF-based monocular thermal-inertial odometry pipeline, run against three thermal datasets and one visible-light baseline with a single shared parameter set.

## Problem

Visual-inertial odometry breaks down in smoke, darkness and low texture — which is exactly where a robot most needs to know where it is. Thermal cameras are a candidate there, because they see emitted heat rather than reflected light. This work builds a monocular thermal-inertial pipeline and finds out where it holds.

The contribution is not a new algorithm. It is a test harness that separates preprocessing, front-end (KLT / ORB) and back-end (MSCKF) into three independent layers, then runs the same filter across several datasets **without retuning anything per dataset** — so differences in performance come from the data, not from parameter fiddling.

## Results

Best run: **KLT front-end, CLAHE on, normalise→CLAHE ordering**, on the SThereo `valley_evening` ground-vehicle sequence — the only sequence that tracked from start to finish. ZUPT is off in everything reported here.

![SThereo trajectory](docs/figures/sthereo_trajectory.png)

*Left: position over time. Right: top-down view after 4-DOF alignment. Ground truth dashed red, estimate blue.*

| | |
|---|---|
| Ground-truth path length | 2014 m |
| ATE RMSE | **96.4 m — 4.8 % of path length** |
| Estimated path length | 1673 m → ratio **0.83**, so scale is short by about 17 % |
| Endpoint error after alignment | 157 m |

Relative pose error from `evo`, divided by segment length:

| Segment length δ | RPE [m/m] |
|---|---|
| 1 m | 0.771 |
| 10 m | 0.448 |
| 50 m | 0.387 |
| 100 m | 0.349 |

> **The 1 m row does not mean what it looks like.** SThereo's ground-truth poses sit 1.30 m apart on average, so a 1 m segment is shorter than the ground truth can actually resolve. That row measures the reference data's coarseness, not the system's per-metre error. Use the 10 / 50 / 100 m rows.

The error per metre keeps dropping as segments get longer, which separates short-range jitter from real drift — over long stretches the error settles instead of growing. Put that together with the 17 % scale shortfall and the picture is clear: **the problem is scale and orientation drift, not the front-end losing track.** The diagnostics back that up.

![SThereo diagnostics](docs/figures/sthereo_diagnostics.png)

*Top: error over time against the RMSE line. Bottom: 508 tracks alive on average, of which about 19 make it through the χ² gate into the filter.*

### The other datasets

| Dataset | What happened |
|---|---|
| **SThereo** (ground vehicle, thermal) | Tracked end to end — the result above |
| **ROVTIO** (indoor UAV, thermal) | Held for stretches, then diverged |
| **FIReStereo** (outdoor UAV, thermal) | Never got tracking going at all |
| **EuRoC** (indoor UAV, visible) | Sanity check on the back-end, not a thermal result |

**AerialTN** is wired into the harness but left out of the results, because of an IMU axis-convention mismatch I could not reconcile with the rest of the pipeline. Reporting a trajectory from a sequence whose IMU frame you do not trust would not mean anything.

## Why the ATE uses a 4-DOF alignment

ATE here is computed after aligning **position and yaw only** — four degrees of freedom — rather than the usual full SE(3) or Sim(3) fit.

The reason is that in a visual-*inertial* system, roll and pitch are observable. The accelerometer sees gravity, which pins both tilt angles to something absolute. Scale is observable the same way. An SE(3) alignment is free to rotate the estimate in roll and pitch to make the error look smaller, which quietly hides real tilt error inside the alignment and flatters the result. Position and yaw are the directions that genuinely are unobservable, so those are the only ones the alignment should be allowed to absorb.

`evo` offers SE(3) and Sim(3) but not this, so I implemented the 4-DOF alignment in `mTIO/evaluation.py` and kept `evo`'s SE(3) APE alongside it purely as a check that mine behaves sensibly on the same trajectories.

## Pipeline

```
Thermal PNG ─┐                          ┌─ KLT optical flow ─┐
             ├─ ThermalDataLoader       │                    │
IMU CSV   ─┐ │  (undistort → Gaussian  ─┤                    │
           └─┤   denoise → percentile   └─ ORB descriptor ───┤
GT        ───┤   normalise → CLAHE)                          │
             │           active / dead feature tracks        │
             │                                               ▼
             │                                   ┌── MSCKF back-end ──┐
             ▼                                   │ predict  (IMU, FEJ)│
  VIOSequencer (chronological IMU + cam events) ►│ augment_state      │
                                                 │ triangulate (GN)   │
                                                 │ parallax + χ² gates│
                                                 │ sequential EKF upd.│
                                                 │ prune cam_states   │
                                                 └────────┬───────────┘
                                                          ▼
                                             {pose, velocity, biases}
```

Two back-end choices worth mentioning:

- **First-Estimate Jacobians.** Every Jacobian is linearised at the state's first estimate, frozen, while the residual uses the current one. This keeps the unobservable directions consistent and kills a fake yaw-drift feedback loop that a plain EKF-MSCKF develops.
- **One track at a time.** Each accepted track updates the covariance on its own instead of being stacked into one big measurement. Batching hundreds of tracks at once shrank the covariance so hard that the next update's χ² gate rejected everything, and the filter collapsed into pure IMU dead reckoning.

## Running it

Setup, settings, repository layout and dataset sources are in [`docs/running.md`](docs/running.md).

## License

MIT — see [LICENSE](LICENSE).
