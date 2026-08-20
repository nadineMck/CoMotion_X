"""Typed project configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    name: str
    seed: int


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    timestep_seconds: float
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class RobotConfig:
    model_path: Path
    move_duration_seconds: float


@dataclass(frozen=True, slots=True)
class HumanConfig:
    scenario: str
    duration_seconds: float
    frames_per_second: float
    noise_standard_deviation_m: float
    dropout_probability: float


@dataclass(frozen=True, slots=True)
class EstimationConfig:
    observation_noise_standard_deviation_m: float
    process_acceleration_standard_deviation_mps2: float
    initial_velocity_standard_deviation_mps: float


@dataclass(frozen=True, slots=True)
class PredictionConfig:
    horizons_seconds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    warning_distance_m: float
    critical_distance_m: float
    slow_velocity_scale: float


@dataclass(frozen=True, slots=True)
class AppConfig:
    project: ProjectConfig
    simulation: SimulationConfig
    robot: RobotConfig
    human: HumanConfig
    estimation: EstimationConfig
    prediction: PredictionConfig
    safety: SafetyConfig


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid [{name}] configuration section")
    return value


def load_config(path: Path | str) -> AppConfig:
    config_path = Path(path)
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)

    project = _section(raw, "project")
    simulation = _section(raw, "simulation")
    robot = _section(raw, "robot")
    human = _section(raw, "human")
    estimation = _section(raw, "estimation")
    prediction = _section(raw, "prediction")
    safety = _section(raw, "safety")

    config = AppConfig(
        project=ProjectConfig(name=str(project["name"]), seed=int(project["seed"])),
        simulation=SimulationConfig(
            timestep_seconds=float(simulation["timestep_seconds"]),
            duration_seconds=float(simulation["duration_seconds"]),
        ),
        robot=RobotConfig(
            model_path=Path(str(robot["model_path"])),
            move_duration_seconds=float(robot["move_duration_seconds"]),
        ),
        human=HumanConfig(
            scenario=str(human["scenario"]),
            duration_seconds=float(human["duration_seconds"]),
            frames_per_second=float(human["frames_per_second"]),
            noise_standard_deviation_m=float(human["noise_standard_deviation_m"]),
            dropout_probability=float(human["dropout_probability"]),
        ),
        estimation=EstimationConfig(
            observation_noise_standard_deviation_m=float(
                estimation["observation_noise_standard_deviation_m"]
            ),
            process_acceleration_standard_deviation_mps2=float(
                estimation["process_acceleration_standard_deviation_mps2"]
            ),
            initial_velocity_standard_deviation_mps=float(
                estimation["initial_velocity_standard_deviation_mps"]
            ),
        ),
        prediction=PredictionConfig(
            horizons_seconds=tuple(float(value) for value in prediction["horizons_seconds"])
        ),
        safety=SafetyConfig(
            warning_distance_m=float(safety["warning_distance_m"]),
            critical_distance_m=float(safety["critical_distance_m"]),
            slow_velocity_scale=float(safety["slow_velocity_scale"]),
        ),
    )
    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    if config.project.seed < 0:
        raise ValueError("project.seed must be non-negative")
    if config.simulation.timestep_seconds <= 0 or config.simulation.duration_seconds <= 0:
        raise ValueError("simulation times must be positive")
    if config.robot.move_duration_seconds <= 0:
        raise ValueError("robot move duration must be positive")
    if config.human.duration_seconds <= 0 or config.human.frames_per_second <= 0:
        raise ValueError("human scenario duration and frame rate must be positive")
    if config.human.noise_standard_deviation_m < 0:
        raise ValueError("human observation noise must be non-negative")
    if not 0 <= config.human.dropout_probability <= 1:
        raise ValueError("human dropout probability must be between 0 and 1")
    if config.estimation.observation_noise_standard_deviation_m <= 0:
        raise ValueError("estimation observation noise must be positive")
    if config.estimation.process_acceleration_standard_deviation_mps2 <= 0:
        raise ValueError("estimation process acceleration noise must be positive")
    if config.estimation.initial_velocity_standard_deviation_mps <= 0:
        raise ValueError("estimation initial velocity uncertainty must be positive")
    if not config.prediction.horizons_seconds:
        raise ValueError("at least one prediction horizon is required")
    if any(value <= 0 for value in config.prediction.horizons_seconds):
        raise ValueError("prediction horizons must be positive")
    if tuple(sorted(config.prediction.horizons_seconds)) != config.prediction.horizons_seconds:
        raise ValueError("prediction horizons must be in ascending order")
    if not 0 < config.safety.critical_distance_m < config.safety.warning_distance_m:
        raise ValueError("critical distance must be positive and smaller than warning distance")
    if not 0 <= config.safety.slow_velocity_scale <= 1:
        raise ValueError("slow velocity scale must be between 0 and 1")
