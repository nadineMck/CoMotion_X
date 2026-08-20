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


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        return run_smoke(args.config)
    raise RuntimeError(f"Unhandled command: {args.command}")
