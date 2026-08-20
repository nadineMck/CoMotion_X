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
from comotion_x.human_model.replay import HumanReplay
from comotion_x.human_model.scenarios import ScenarioName, generate_scenario
from comotion_x.robot.simulation import PandaSimulation


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
    raise RuntimeError(f"Unhandled command: {args.command}")
