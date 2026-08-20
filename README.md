# CoMotion-X

**Uncertainty-Aware Predictive Human–Robot Collaboration**

CoMotion-X is a research prototype for comparing reactive robot-safety control with
uncertainty-aware prediction of short-horizon human motion.

The project is currently at **M0: reproducible project skeleton**. See
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
safety state. It does not start a simulator; MuJoCo arrives in M1.

## Current structure

```text
config/                 Experiment configuration
src/comotion_x/core/    Shared types, configuration, logging, reproducibility
tests/                  Automated tests
data/                   Input and generated scenario data
results/                Machine-readable runs, plots, and tables
```
