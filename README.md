# CoMotion-X

**Uncertainty-Aware Predictive Human–Robot Collaboration**

CoMotion-X is a research prototype for comparing reactive robot-safety control with
uncertainty-aware prediction of short-horizon human motion.

The project is currently at **M7: laptop-camera and recorded-video pose integration**. See
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

## Camera and recorded-video testing

Download and verify the pinned MediaPipe Pose Landmarker Lite model:

```bash
uv run comotion-x prepare-camera-model --config config/default.toml
```

Run the laptop webcam continuously with a live skeleton and safety-status overlay:

```bash
uv run comotion-x camera \
  --config config/default.toml \
  --device 0 \
  --display \
  --record-poses data/raw/webcam_session.json
```

The session has no time limit when `--duration` is omitted. Press `q`, close the window, or
use `Ctrl+C` to stop it. Add `--duration 30` when you want a timed 30-second test. On macOS,
grant the terminal or Codex application camera
access under **System Settings → Privacy & Security → Camera** before the first run.

Use a recorded video for reproducible perception tests:

```bash
uv run comotion-x camera \
  --config config/default.toml \
  --video data/raw/test_video.mp4 \
  --record-poses data/processed/test_video_poses.json
```

MediaPipe's body-relative 3D pose is converted into the MuJoCo world frame using
`config/camera_to_world.json`. Its included transform is an initial demonstration alignment;
translation and orientation must be calibrated for the physical camera position before using
measured distances as experimental results.

### Synchronized 3D robot visualization on macOS

MuJoCo's passive viewer on macOS must run through the included `mjpython` launcher. To see
the camera overlay and 3D Panda/digital-human scene at the same time, run:

```bash
.venv/bin/mjpython -m comotion_x camera \
  --config config/default.toml \
  --device 0 \
  --display \
  --robot-display \
  --record-poses data/raw/webcam_visualization.json
```

This command runs until you close it. The red markers are the tracked human joints. The translucent yellow sphere sequence shows
the predicted right-wrist positions at 100, 200, 300, and 500 ms; sphere size represents
prediction uncertainty. Closing the MuJoCo window or pressing `q` in the camera window ends
the session.

## Current structure

```text
config/                 Experiment configuration
src/comotion_x/core/    Shared types, configuration, logging, reproducibility
src/comotion_x/estimation/  Kalman filtering and human motion-state estimation
src/comotion_x/evaluation/  Prediction metrics and experiment helpers
src/comotion_x/human_model/  Human scenario generation and timestamped replay
src/comotion_x/prediction/  Short-horizon prediction and covariance propagation
src/comotion_x/perception/  Camera/video capture, calibration, and MediaPipe pose
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
