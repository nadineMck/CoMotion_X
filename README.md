# CoMotion-X

**Uncertainty-Aware Predictive Human–Robot Collaboration**

CoMotion-X is a research prototype for comparing reactive robot-safety control with
uncertainty-aware prediction of short-horizon human motion.

The project is currently at **M6: reproducible experiments and quantitative results**. See
[`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the complete roadmap.

## Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/) (recommended)

## Setup and validation

```bash
uv sync --extra dev
uv run comotion-x smoke --config config/default.toml
uv run pytest
uv run ruff check .
```

For a traditional `pip` setup, install `requirements.txt`. Copy `.env.example` to `.env`
when setting up a new clone; the local `.env` is ignored by Git.

The smoke command prints one JSON record containing the configuration, seed, and initial
safety state.

Run the headless Franka Panda reaching baseline:

```bash
uv run comotion-x simulate --config config/default.toml --duration 4
```

The command repeatedly moves between two safe joint configurations and reports the simulated
duration, completed movements, end-effector path length, final tracking error, and final pose.

Generate all eight reproducible human-motion datasets:

```bash
uv run comotion-x generate-scenarios --config config/default.toml --output data/scenarios
```

Replay one scenario with the moving Panda in the shared MuJoCo world frame:

```bash
uv run comotion-x replay-human --config config/default.toml --scenario crossing
```

Available scenarios are `stationary`, `slow_approach`, `sudden_reach`, `crossing`,
`near_miss`, `withdrawal`, `occlusion`, and `variable_speed`. Each exported file contains
clean ground-truth frames and separate noisy/dropout-affected observation frames.

Evaluate uncertainty-aware right-wrist predictions at the configured horizons:

```bash
uv run comotion-x evaluate-prediction \
  --config config/default.toml \
  --scenario variable_speed
```

The estimator tracks 3D position and velocity with a per-joint constant-velocity Kalman
filter. During missing observations it propagates the state forward and expands covariance.
The evaluator reports prediction error, RMSE, empirical 95% uncertainty coverage, and mean
uncertainty radius at 100, 200, 300, and 500 ms.

Evaluate predicted human–robot clearance over an entire scenario:

```bash
uv run comotion-x evaluate-risk \
  --config config/default.toml \
  --scenario crossing
```

Human wrists and torso are represented by spheres, arms by capsules, and the Panda by
capsules along every major link plus a hand sphere. Human volumes expand from predicted
covariance. The risk engine aligns human and robot trajectories by timestamp and reports the
minimum signed clearance, time to closest approach, and closest primitive pair. Negative
clearance means the occupancy volumes overlap.

Compare the unaware, current-clearance reactive, and future-clearance predictive controllers:

```bash
uv run comotion-x compare-controllers \
  --config config/default.toml \
  --scenario crossing
```

The controller state machine uses `NORMAL`, `CAUTION`, `HIGH_RISK`, and `CRITICAL` modes.
These map to full speed, two configurable slowdown levels, and controlled stop. Escalation is
immediate; hysteresis and minimum dwell time prevent rapid de-escalation and mode oscillation.
The comparison replays identical human ground truth for all policies and reports intervention
time, stop time, actual minimum clearance, idle time, and robot task progress.

Run the complete experiment matrix and regenerate all quantitative artifacts:

```bash
uv run comotion-x run-experiments \
  --config config/default.toml \
  --seeds 42 \
  --output results/runs/latest
```

Use comma-separated seeds such as `--seeds 42,43,44` for repeated noisy trials. The default
suite evaluates all eight scenarios using unaware, reactive, deterministic-predictive, and
uncertainty-aware predictive controllers. It creates:

- `trial_metrics.csv` — one safety/productivity summary per trial;
- `timesteps.csv` — every controller decision and clearance measurement;
- `prediction_horizons.csv` — prediction error and calibration by horizon;
- `controller_summary.csv` — aggregate controller comparison;
- `manifest.json` — seeds, scenarios, controllers, horizons, and record counts;
- `figures/controller_summary.png` and `figures/scenario_clearance.png`.

Generated results are intentionally ignored by Git and can be reproduced from the committed
configuration and command.

## Current structure

```text
config/                 Experiment configuration
src/comotion_x/core/    Shared types, configuration, logging, reproducibility
src/comotion_x/estimation/  Kalman filtering and human motion-state estimation
src/comotion_x/evaluation/  Prediction metrics and experiment helpers
src/comotion_x/human_model/  Human scenario generation and timestamped replay
src/comotion_x/prediction/  Short-horizon prediction and covariance propagation
src/comotion_x/robot/   MuJoCo state, trajectory, and reaching simulation
src/comotion_x/safety/  Occupancy geometry and spatiotemporal collision risk
tests/                  Automated tests
data/                   Input and generated scenario data
results/                Machine-readable runs, plots, and tables
```

## Third-party robot model

The Franka Emika Panda MJCF and mesh assets are imported from the official
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). The exact upstream
revision and license are stored in `third_party/mujoco_menagerie/`.
