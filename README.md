# CoMotion-X

**Uncertainty-Aware Predictive Human–Robot Collaboration**

CoMotion-X is a research prototype for comparing reactive robot-safety control with
uncertainty-aware prediction of short-horizon human motion.

The project is currently at **M3: filtered human-state estimation and prediction**. See
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

## Current structure

```text
config/                 Experiment configuration
src/comotion_x/core/    Shared types, configuration, logging, reproducibility
src/comotion_x/estimation/  Kalman filtering and human motion-state estimation
src/comotion_x/evaluation/  Prediction metrics and experiment helpers
src/comotion_x/human_model/  Human scenario generation and timestamped replay
src/comotion_x/prediction/  Short-horizon prediction and covariance propagation
src/comotion_x/robot/   MuJoCo state, trajectory, and reaching simulation
tests/                  Automated tests
data/                   Input and generated scenario data
results/                Machine-readable runs, plots, and tables
```

## Third-party robot model

The Franka Emika Panda MJCF and mesh assets are imported from the official
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). The exact upstream
revision and license are stored in `third_party/mujoco_menagerie/`.
