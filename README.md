# CoMotion-X

**Uncertainty-Aware Predictive Human–Robot Collaboration**

CoMotion-X is a research prototype for comparing reactive robot-safety control with
uncertainty-aware prediction of short-horizon human motion.

The project is currently at **M1: MuJoCo robot simulation baseline**. See
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

## Current structure

```text
config/                 Experiment configuration
src/comotion_x/core/    Shared types, configuration, logging, reproducibility
src/comotion_x/robot/   MuJoCo state, trajectory, and reaching simulation
tests/                  Automated tests
data/                   Input and generated scenario data
results/                Machine-readable runs, plots, and tables
```

## Third-party robot model

The Franka Emika Panda MJCF and mesh assets are imported from the official
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). The exact upstream
revision and license are stored in `third_party/mujoco_menagerie/`.
