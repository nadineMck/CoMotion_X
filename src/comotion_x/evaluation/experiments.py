"""Reproducible batch experiments, tables, logs, and figures."""

from __future__ import annotations

import csv
import json
import os
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "comotion-x-matplotlib-cache")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from comotion_x.core.config import AppConfig
from comotion_x.evaluation.controllers import run_controller_trial
from comotion_x.evaluation.prediction import evaluate_scenario_prediction
from comotion_x.human_model.scenarios import ScenarioName, generate_scenario

DEFAULT_CONTROLLERS = (
    "unaware",
    "reactive",
    "predictive_deterministic",
    "predictive_uncertainty",
)


@dataclass(frozen=True, slots=True)
class ExperimentArtifacts:
    output_directory: Path
    trial_count: int
    timestep_count: int
    files: tuple[Path, ...]


def run_experiment_suite(
    config: AppConfig,
    *,
    seeds: tuple[int, ...],
    output_directory: Path | str,
    scenario_names: tuple[ScenarioName, ...] = tuple(ScenarioName),
    controller_names: tuple[str, ...] = DEFAULT_CONTROLLERS,
    include_prediction_study: bool = True,
) -> ExperimentArtifacts:
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("experiments require at least one non-negative seed")
    output = Path(output_directory)
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    trial_rows: list[dict[str, Any]] = []
    timestep_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for seed in seeds:
        for scenario_name in scenario_names:
            scenario = generate_scenario(
                scenario_name,
                duration_seconds=config.human.duration_seconds,
                frames_per_second=config.human.frames_per_second,
                seed=seed,
                noise_standard_deviation_m=config.human.noise_standard_deviation_m,
                dropout_probability=config.human.dropout_probability,
            )
            for controller_name in controller_names:
                result = run_controller_trial(controller_name, scenario, config)
                row: dict[str, Any] = {
                    "seed": seed,
                    "scenario": scenario_name.value,
                    "controller": controller_name,
                    **result.metrics.as_dict(),
                }
                trial_rows.append(row)
                timestep_rows.extend(
                    {
                        "seed": seed,
                        "scenario": scenario_name.value,
                        "controller": controller_name,
                        "timestamp": round(item.timestamp, 9),
                        "mode": item.mode.value,
                        "velocity_scale": item.velocity_scale,
                        "actual_clearance_m": item.actual_clearance_m,
                        "measured_clearance_m": item.measured_clearance_m,
                        "predicted_clearance_m": item.predicted_clearance_m,
                        "task_progress_seconds": item.task_progress_seconds,
                    }
                    for item in result.timesteps
                )

            if include_prediction_study:
                prediction_metrics = evaluate_scenario_prediction(
                    scenario,
                    config.prediction.horizons_seconds,
                    observation_std_m=(
                        config.estimation.observation_noise_standard_deviation_m
                    ),
                    acceleration_std_mps2=(
                        config.estimation.process_acceleration_standard_deviation_mps2
                    ),
                    initial_velocity_std_mps=(
                        config.estimation.initial_velocity_standard_deviation_mps
                    ),
                )
                prediction_rows.extend(
                    {
                        "seed": seed,
                        "scenario": scenario_name.value,
                        "horizon_seconds": metric.horizon_seconds,
                        **metric.as_dict(),
                    }
                    for metric in prediction_metrics
                )

    _add_comparative_metrics(trial_rows)
    summary_rows = _summarize_trials(trial_rows)
    files = (
        output / "trial_metrics.csv",
        output / "timesteps.csv",
        output / "prediction_horizons.csv",
        output / "controller_summary.csv",
        output / "manifest.json",
        figures / "controller_summary.png",
        figures / "scenario_clearance.png",
    )
    _write_csv(files[0], trial_rows)
    _write_csv(files[1], timestep_rows)
    _write_csv(files[2], prediction_rows)
    _write_csv(files[3], summary_rows)
    files[4].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seeds": list(seeds),
                "scenarios": [name.value for name in scenario_names],
                "controllers": list(controller_names),
                "prediction_horizons_seconds": list(config.prediction.horizons_seconds),
                "trial_count": len(trial_rows),
                "timestep_count": len(timestep_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _plot_controller_summary(summary_rows, files[5])
    _plot_scenario_clearance(trial_rows, files[6])
    return ExperimentArtifacts(
        output_directory=output,
        trial_count=len(trial_rows),
        timestep_count=len(timestep_rows),
        files=files,
    )


def _add_comparative_metrics(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["scenario"]))].append(row)
    for group in grouped.values():
        by_controller = {str(row["controller"]): row for row in group}
        unaware_clearance = float(by_controller["unaware"]["minimum_actual_clearance_m"])
        reactive_time = by_controller.get("reactive", {}).get("first_intervention_timestamp")
        for row in group:
            stop_count = int(row["stop_count"])
            row["unnecessary_stop_count"] = stop_count if unaware_clearance > 0 else 0
            intervention_time = row.get("first_intervention_timestamp")
            row["intervention_lead_seconds"] = (
                round(float(reactive_time) - float(intervention_time), 6)
                if reactive_time is not None and intervention_time is not None
                else None
            )


def _summarize_trials(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["controller"])].append(row)
    summaries: list[dict[str, Any]] = []
    for controller, controller_rows in sorted(grouped.items()):
        summaries.append(
            {
                "controller": controller,
                "trials": len(controller_rows),
                "mean_minimum_clearance_m": statistics.fmean(
                    float(row["minimum_actual_clearance_m"]) for row in controller_rows
                ),
                "total_safety_violations": sum(
                    int(row["safety_violation_count"]) for row in controller_rows
                ),
                "total_stops": sum(int(row["stop_count"]) for row in controller_rows),
                "total_unnecessary_stops": sum(
                    int(row["unnecessary_stop_count"]) for row in controller_rows
                ),
                "mean_idle_time_seconds": statistics.fmean(
                    float(row["idle_time_seconds"]) for row in controller_rows
                ),
                "mean_productivity_ratio": statistics.fmean(
                    float(row["productivity_ratio"]) for row in controller_rows
                ),
                "mean_intervention_lead_seconds": _mean_optional(
                    row["intervention_lead_seconds"] for row in controller_rows
                ),
            }
        )
    return summaries


def _mean_optional(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_controller_summary(rows: list[dict[str, Any]], path: Path) -> None:
    controllers = [str(row["controller"]) for row in rows]
    clearance = [float(row["mean_minimum_clearance_m"]) for row in rows]
    productivity = [float(row["mean_productivity_ratio"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(controllers, clearance, color="#2f6f9f")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Mean minimum clearance (m)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(controllers, productivity, color="#e17c45")
    axes[1].set_ylabel("Mean task-progress ratio")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=20)
    figure.suptitle("CoMotion-X controller safety–productivity comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_scenario_clearance(rows: list[dict[str, Any]], path: Path) -> None:
    scenarios = sorted({str(row["scenario"]) for row in rows})
    controllers = sorted({str(row["controller"]) for row in rows})
    width = 0.8 / len(controllers)
    figure, axis = plt.subplots(figsize=(12, 5))
    for index, controller in enumerate(controllers):
        values = [
            statistics.fmean(
                float(row["minimum_actual_clearance_m"])
                for row in rows
                if row["scenario"] == scenario and row["controller"] == controller
            )
            for scenario in scenarios
        ]
        positions = [item + index * width for item in range(len(scenarios))]
        axis.bar(positions, values, width=width, label=controller)
    center_offset = width * (len(controllers) - 1) / 2
    axis.set_xticks([item + center_offset for item in range(len(scenarios))], scenarios)
    axis.tick_params(axis="x", rotation=25)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Mean minimum clearance (m)")
    axis.set_title("Minimum clearance by scenario and controller")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
