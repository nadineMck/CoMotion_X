"""Command-line interface for CoMotion-X."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from comotion_x.core.config import load_config
from comotion_x.core.logging import emit_event
from comotion_x.core.models import RiskAssessment, SafetyMode
from comotion_x.core.reproducibility import seed_everything
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


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        return run_smoke(args.config)
    if args.command == "simulate":
        return run_simulation(args.config, args.duration)
    raise RuntimeError(f"Unhandled command: {args.command}")
