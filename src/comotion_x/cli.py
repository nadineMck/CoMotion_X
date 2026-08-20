"""Command-line interface for CoMotion-X."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from comotion_x.core.config import load_config
from comotion_x.core.logging import emit_event
from comotion_x.core.models import RiskAssessment, SafetyMode
from comotion_x.core.reproducibility import seed_everything
from comotion_x.estimation.state_estimator import HumanStateEstimator
from comotion_x.evaluation.controllers import compare_controllers
from comotion_x.evaluation.experiments import run_experiment_suite
from comotion_x.evaluation.prediction import evaluate_scenario_prediction
from comotion_x.human_model.replay import HumanReplay
from comotion_x.human_model.scenarios import ScenarioName, generate_scenario
from comotion_x.perception.calibration import CameraWorldTransform
from comotion_x.perception.camera import OpenCVFrameSource
from comotion_x.perception.live import run_live_camera
from comotion_x.perception.model import download_pose_model, verify_pose_model
from comotion_x.perception.pose_estimator import MediaPipePoseEstimator
from comotion_x.prediction.motion_predictor import HumanMotionPredictor
from comotion_x.robot.simulation import PandaSimulation
from comotion_x.safety.occupancy import OccupancyParameters
from comotion_x.safety.risk import CollisionRiskEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="comotion-x", description="CoMotion-X research tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="validate the M0 project foundation")
    smoke.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.toml"),
        help="path to a TOML configuration file",
    )

    simulate = subparsers.add_parser("simulate", help="run the M1 Franka reaching demo")
    simulate.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.toml"),
        help="path to a TOML configuration file",
    )
    simulate.add_argument(
        "--duration",
        type=float,
        default=None,
        help="override simulation duration in seconds",
    )

    generate = subparsers.add_parser(
        "generate-scenarios", help="export deterministic M2 human scenarios"
    )
    generate.add_argument("--config", type=Path, default=Path("config/default.toml"))
    generate.add_argument("--output", type=Path, default=Path("data/scenarios"))

    replay = subparsers.add_parser(
        "replay-human", help="replay a 3D human scenario with the Franka simulation"
    )
    replay.add_argument("--config", type=Path, default=Path("config/default.toml"))
    replay.add_argument("--scenario", choices=[name.value for name in ScenarioName], default=None)

    evaluate = subparsers.add_parser(
        "evaluate-prediction", help="evaluate M3 wrist prediction by horizon"
    )
    evaluate.add_argument("--config", type=Path, default=Path("config/default.toml"))
    evaluate.add_argument(
        "--scenario", choices=[name.value for name in ScenarioName], default="variable_speed"
    )

    risk = subparsers.add_parser(
        "evaluate-risk", help="evaluate M4 predicted human-robot geometric clearance"
    )
    risk.add_argument("--config", type=Path, default=Path("config/default.toml"))
    risk.add_argument(
        "--scenario", choices=[name.value for name in ScenarioName], default="crossing"
    )

    compare = subparsers.add_parser(
        "compare-controllers", help="compare unaware, reactive, and predictive M5 policies"
    )
    compare.add_argument("--config", type=Path, default=Path("config/default.toml"))
    compare.add_argument(
        "--scenario", choices=[name.value for name in ScenarioName], default="crossing"
    )

    experiments = subparsers.add_parser(
        "run-experiments", help="run the reproducible M6 experiment suite"
    )
    experiments.add_argument("--config", type=Path, default=Path("config/default.toml"))
    experiments.add_argument(
        "--seeds", default="42", help="comma-separated non-negative random seeds"
    )
    experiments.add_argument("--output", type=Path, default=Path("results/runs/latest"))

    prepare_camera = subparsers.add_parser(
        "prepare-camera-model", help="download and verify the pinned MediaPipe pose model"
    )
    prepare_camera.add_argument("--config", type=Path, default=Path("config/default.toml"))

    camera = subparsers.add_parser(
        "camera", help="run live or recorded camera pose control with the simulated Panda"
    )
    camera.add_argument("--config", type=Path, default=Path("config/default.toml"))
    source_group = camera.add_mutually_exclusive_group()
    source_group.add_argument("--device", type=int, default=None, help="webcam device index")
    source_group.add_argument("--video", type=Path, default=None, help="recorded video path")
    camera.add_argument("--duration", type=float, default=None)
    camera.add_argument("--display", action="store_true", help="show the camera overlay; q exits")
    camera.add_argument("--record-poses", type=Path, default=None)
    return parser


def run_smoke(config_path: Path) -> int:
    config = load_config(config_path)
    seed_everything(config.project.seed)
    initial_risk = RiskAssessment(
        timestamp=0.0,
        min_distance_m=float("inf"),
        time_to_closest_seconds=None,
        collision_probability=0.0,
        mode=SafetyMode.NORMAL,
    )
    emit_event(
        "smoke_check_passed",
        project=config.project.name,
        seed=config.project.seed,
        timestep_seconds=config.simulation.timestep_seconds,
        prediction_horizons_seconds=config.prediction.horizons_seconds,
        safety_mode=initial_risk.mode.value,
    )
    return 0


def run_simulation(config_path: Path, duration: float | None) -> int:
    config = load_config(config_path)
    seed_everything(config.project.seed)
    simulation = PandaSimulation(
        model_path=config.robot.model_path,
        control_timestep_seconds=config.simulation.timestep_seconds,
        move_duration_seconds=config.robot.move_duration_seconds,
    )
    summary = simulation.run(
        duration_seconds=duration or config.simulation.duration_seconds,
    )
    emit_event("simulation_completed", **summary.as_dict())
    return 0


def _scenario_from_config(config, scenario_name: str):
    return generate_scenario(
        scenario_name,
        duration_seconds=config.human.duration_seconds,
        frames_per_second=config.human.frames_per_second,
        seed=config.project.seed,
        noise_standard_deviation_m=config.human.noise_standard_deviation_m,
        dropout_probability=config.human.dropout_probability,
    )


def run_generate_scenarios(config_path: Path, output: Path) -> int:
    config = load_config(config_path)
    exported: list[str] = []
    for scenario_name in ScenarioName:
        scenario = _scenario_from_config(config, scenario_name.value)
        path = output / f"{scenario_name.value}.json"
        scenario.export_json(path)
        exported.append(str(path))
    emit_event("human_scenarios_generated", count=len(exported), files=exported)
    return 0


def run_human_replay(config_path: Path, scenario_name: str | None) -> int:
    config = load_config(config_path)
    selected_name = scenario_name or config.human.scenario
    scenario = _scenario_from_config(config, selected_name)
    replay = HumanReplay(scenario)
    simulation = PandaSimulation(
        model_path=config.robot.model_path,
        control_timestep_seconds=config.simulation.timestep_seconds,
        move_duration_seconds=config.robot.move_duration_seconds,
    )
    steps = math.ceil(scenario.duration_seconds / config.simulation.timestep_seconds)
    observed_source_frames: set[float] = set()
    missing_observations = 0
    for _ in range(steps):
        frame = replay.frame_at(float(simulation.data.time))
        observed_source_frames.add(frame.timestamp)
        missing_observations += len(simulation.human_marker_names) - len(frame.joints)
        simulation.set_human_pose(frame)
        simulation.step()
    final_frame = replay.frame_at(scenario.duration_seconds)
    emit_event(
        "human_replay_completed",
        scenario=scenario.name.value,
        frame_id=final_frame.frame_id,
        duration_seconds=round(float(simulation.data.time), 6),
        simulation_steps=steps,
        source_frames_replayed=len(observed_source_frames),
        marker_count=len(simulation.human_marker_names),
        missing_joint_observations=missing_observations,
        final_right_wrist_m=list(final_frame.joints["right_wrist"].position_m),
    )
    return 0


def run_prediction_evaluation(config_path: Path, scenario_name: str) -> int:
    config = load_config(config_path)
    scenario = _scenario_from_config(config, scenario_name)
    metrics = evaluate_scenario_prediction(
        scenario,
        config.prediction.horizons_seconds,
        observation_std_m=config.estimation.observation_noise_standard_deviation_m,
        acceleration_std_mps2=(
            config.estimation.process_acceleration_standard_deviation_mps2
        ),
        initial_velocity_std_mps=config.estimation.initial_velocity_standard_deviation_mps,
    )
    emit_event(
        "prediction_evaluation_completed",
        scenario=scenario.name.value,
        joint="right_wrist",
        horizons={str(metric.horizon_seconds): metric.as_dict() for metric in metrics},
    )
    return 0


def _occupancy_parameters(config) -> OccupancyParameters:
    return OccupancyParameters(
        human_wrist_radius_m=config.occupancy.human_wrist_radius_m,
        human_arm_radius_m=config.occupancy.human_arm_radius_m,
        human_torso_radius_m=config.occupancy.human_torso_radius_m,
        robot_link_radius_m=config.occupancy.robot_link_radius_m,
        robot_hand_radius_m=config.occupancy.robot_hand_radius_m,
        uncertainty_sigma=config.occupancy.uncertainty_sigma,
    )


def run_risk_evaluation(config_path: Path, scenario_name: str) -> int:
    config = load_config(config_path)
    scenario = _scenario_from_config(config, scenario_name)
    estimator = HumanStateEstimator(
        observation_std_m=config.estimation.observation_noise_standard_deviation_m,
        acceleration_std_mps2=config.estimation.process_acceleration_standard_deviation_mps2,
        initial_velocity_std_mps=config.estimation.initial_velocity_standard_deviation_mps,
    )
    predictor = HumanMotionPredictor(
        acceleration_std_mps2=config.estimation.process_acceleration_standard_deviation_mps2
    )
    simulation = PandaSimulation(
        model_path=config.robot.model_path,
        control_timestep_seconds=config.simulation.timestep_seconds,
        move_duration_seconds=config.robot.move_duration_seconds,
    )
    engine = CollisionRiskEngine(_occupancy_parameters(config))
    closest_risk = None
    closest_source_timestamp = 0.0
    assessments = 0

    for frame in scenario.observation_frames:
        state = estimator.update(frame)
        if "right_wrist" not in state.joints or frame.timestamp < 0.5:
            continue
        prediction = predictor.predict(state, config.prediction.horizons_seconds)
        robot_trajectory = simulation.planned_link_trajectory(
            tuple(prediction_slice.timestamp for prediction_slice in prediction.slices)
        )
        assessment = engine.assess(prediction, robot_trajectory)
        assessments += 1
        if (
            closest_risk is None
            or assessment.minimum_clearance_m < closest_risk.minimum_clearance_m
        ):
            closest_risk = assessment
            closest_source_timestamp = frame.timestamp

    if closest_risk is None:
        raise RuntimeError("scenario produced no risk assessments")
    closest_slice = min(closest_risk.slices, key=lambda item: item.clearance_m)
    emit_event(
        "risk_evaluation_completed",
        scenario=scenario.name.value,
        assessments=assessments,
        minimum_clearance_m=round(closest_risk.minimum_clearance_m, 6),
        collision_predicted=closest_risk.collision_predicted,
        source_timestamp=round(closest_source_timestamp, 6),
        time_to_closest_seconds=closest_risk.time_to_closest_seconds,
        closest_human_primitive=closest_slice.human_primitive,
        closest_robot_primitive=closest_slice.robot_primitive,
    )
    return 0


def run_controller_comparison(config_path: Path, scenario_name: str) -> int:
    config = load_config(config_path)
    scenario = _scenario_from_config(config, scenario_name)
    metrics = compare_controllers(scenario, config)
    metrics_by_name = {metric.controller: metric for metric in metrics}
    reactive_time = metrics_by_name["reactive"].first_intervention_timestamp
    predictive_time = metrics_by_name["predictive"].first_intervention_timestamp
    lead_time = (
        reactive_time - predictive_time
        if reactive_time is not None and predictive_time is not None
        else None
    )
    emit_event(
        "controller_comparison_completed",
        scenario=scenario.name.value,
        controllers={metric.controller: metric.as_dict() for metric in metrics},
        predictive_intervention_lead_seconds=(
            round(lead_time, 6) if lead_time is not None else None
        ),
    )
    return 0


def run_experiments(config_path: Path, seeds_text: str, output: Path) -> int:
    config = load_config(config_path)
    try:
        seeds = tuple(int(value.strip()) for value in seeds_text.split(",") if value.strip())
    except ValueError as error:
        raise ValueError("--seeds must contain comma-separated integers") from error
    artifacts = run_experiment_suite(config, seeds=seeds, output_directory=output)
    emit_event(
        "experiment_suite_completed",
        output_directory=str(artifacts.output_directory),
        trial_count=artifacts.trial_count,
        timestep_count=artifacts.timestep_count,
        files=[str(path) for path in artifacts.files],
    )
    return 0


def prepare_camera_model(config_path: Path) -> int:
    config = load_config(config_path)
    if not verify_pose_model(config.camera.model_path):
        download_pose_model(config.camera.model_path)
    emit_event(
        "camera_model_ready",
        model_path=str(config.camera.model_path),
        verified=verify_pose_model(config.camera.model_path),
    )
    return 0


def run_camera(
    config_path: Path,
    *,
    device: int | None,
    video: Path | None,
    duration: float | None,
    display: bool,
    record_poses: Path | None,
) -> int:
    config = load_config(config_path)
    if not verify_pose_model(config.camera.model_path):
        raise FileNotFoundError("camera model is not ready; run `comotion-x prepare-camera-model`")
    transform = CameraWorldTransform.from_json(config.camera.calibration_path)
    source_value = (
        video
        if video is not None
        else (device if device is not None else config.camera.device_id)
    )
    maximum_duration = duration or config.camera.maximum_duration_seconds
    if maximum_duration <= 0:
        raise ValueError("camera duration must be positive")
    with OpenCVFrameSource(
        source_value,
        width=config.camera.width,
        height=config.camera.height,
    ) as source:
        with MediaPipePoseEstimator(
            config.camera.model_path,
            transform,
            minimum_confidence=config.camera.minimum_landmark_confidence,
        ) as detector:
            summary = run_live_camera(
                config,
                source,
                detector,
                maximum_duration_seconds=maximum_duration,
                display=display,
                recorded_pose_path=record_poses,
            )
    emit_event("camera_session_completed", **summary.as_dict())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        return run_smoke(args.config)
    if args.command == "simulate":
        return run_simulation(args.config, args.duration)
    if args.command == "generate-scenarios":
        return run_generate_scenarios(args.config, args.output)
    if args.command == "replay-human":
        return run_human_replay(args.config, args.scenario)
    if args.command == "evaluate-prediction":
        return run_prediction_evaluation(args.config, args.scenario)
    if args.command == "evaluate-risk":
        return run_risk_evaluation(args.config, args.scenario)
    if args.command == "compare-controllers":
        return run_controller_comparison(args.config, args.scenario)
    if args.command == "run-experiments":
        return run_experiments(args.config, args.seeds, args.output)
    if args.command == "prepare-camera-model":
        return prepare_camera_model(args.config)
    if args.command == "camera":
        return run_camera(
            args.config,
            device=args.device,
            video=args.video,
            duration=args.duration,
            display=args.display,
            record_poses=args.record_poses,
        )
    raise RuntimeError(f"Unhandled command: {args.command}")
