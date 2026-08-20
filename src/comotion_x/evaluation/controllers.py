"""Run controller baselines on identical deterministic human scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from comotion_x.core.config import AppConfig
from comotion_x.core.models import SafetyMode
from comotion_x.estimation.state_estimator import HumanStateEstimator
from comotion_x.human_model.scenarios import HumanScenario
from comotion_x.prediction.motion_predictor import HumanMotionPredictor
from comotion_x.robot.simulation import PandaSimulation
from comotion_x.safety.controller import (
    ControllerParameters,
    NoAwarenessController,
    PredictiveSafetyController,
    ReactiveSafetyController,
    SafetyController,
)
from comotion_x.safety.occupancy import OccupancyParameters, current_clearance
from comotion_x.safety.risk import CollisionRiskEngine


@dataclass(frozen=True, slots=True)
class ControllerMetrics:
    controller: str
    minimum_actual_clearance_m: float
    intervention_count: int
    stop_count: int
    idle_time_seconds: float
    final_task_progress_seconds: float
    final_mode: SafetyMode
    first_intervention_timestamp: float | None
    first_stop_timestamp: float | None

    def as_dict(self) -> dict[str, str | float | int | None]:
        return {
            "minimum_actual_clearance_m": round(self.minimum_actual_clearance_m, 6),
            "intervention_count": self.intervention_count,
            "stop_count": self.stop_count,
            "idle_time_seconds": round(self.idle_time_seconds, 6),
            "final_task_progress_seconds": round(self.final_task_progress_seconds, 6),
            "final_mode": self.final_mode.value,
            "first_intervention_timestamp": (
                round(self.first_intervention_timestamp, 6)
                if self.first_intervention_timestamp is not None
                else None
            ),
            "first_stop_timestamp": (
                round(self.first_stop_timestamp, 6)
                if self.first_stop_timestamp is not None
                else None
            ),
        }


def controller_parameters(config: AppConfig) -> ControllerParameters:
    return ControllerParameters(
        caution_clearance_m=config.safety.caution_clearance_m,
        high_risk_clearance_m=config.safety.high_risk_clearance_m,
        critical_clearance_m=config.safety.critical_clearance_m,
        caution_velocity_scale=config.safety.caution_velocity_scale,
        high_risk_velocity_scale=config.safety.high_risk_velocity_scale,
        hysteresis_m=config.safety.hysteresis_m,
        minimum_dwell_seconds=config.safety.minimum_dwell_seconds,
    )


def occupancy_parameters(config: AppConfig) -> OccupancyParameters:
    return OccupancyParameters(
        human_wrist_radius_m=config.occupancy.human_wrist_radius_m,
        human_arm_radius_m=config.occupancy.human_arm_radius_m,
        human_torso_radius_m=config.occupancy.human_torso_radius_m,
        robot_link_radius_m=config.occupancy.robot_link_radius_m,
        robot_hand_radius_m=config.occupancy.robot_hand_radius_m,
        uncertainty_sigma=config.occupancy.uncertainty_sigma,
    )


def compare_controllers(
    scenario: HumanScenario, config: AppConfig
) -> tuple[ControllerMetrics, ...]:
    parameters = controller_parameters(config)
    controllers: tuple[tuple[str, SafetyController], ...] = (
        ("unaware", NoAwarenessController()),
        ("reactive", ReactiveSafetyController(parameters)),
        ("predictive", PredictiveSafetyController(parameters)),
    )
    return tuple(
        _run_controller(name, controller, scenario, config) for name, controller in controllers
    )


def _run_controller(
    name: str,
    controller: SafetyController,
    scenario: HumanScenario,
    config: AppConfig,
) -> ControllerMetrics:
    occupancy = occupancy_parameters(config)
    simulation = PandaSimulation(
        model_path=config.robot.model_path,
        control_timestep_seconds=config.simulation.timestep_seconds,
        move_duration_seconds=config.robot.move_duration_seconds,
    )
    estimator = HumanStateEstimator(
        observation_std_m=config.estimation.observation_noise_standard_deviation_m,
        acceleration_std_mps2=config.estimation.process_acceleration_standard_deviation_mps2,
        initial_velocity_std_mps=config.estimation.initial_velocity_standard_deviation_mps,
    )
    predictor = HumanMotionPredictor(
        acceleration_std_mps2=config.estimation.process_acceleration_standard_deviation_mps2
    )
    risk_engine = CollisionRiskEngine(occupancy)
    velocity_scale = 1.0
    previous_mode = SafetyMode.NORMAL
    interventions = 0
    stops = 0
    first_intervention_timestamp = None
    first_stop_timestamp = None
    idle_time = 0.0
    minimum_actual_clearance = float("inf")

    for observation_frame, truth_frame in zip(
        scenario.observation_frames, scenario.ground_truth_frames, strict=True
    ):
        while simulation.data.time < truth_frame.timestamp - 1e-9:
            simulation.set_human_pose(truth_frame)
            simulation.step(velocity_scale)
            if velocity_scale == 0:
                idle_time += config.simulation.timestep_seconds

        robot_state = simulation.state()
        actual_clearance = current_clearance(truth_frame, robot_state, occupancy)
        measured_clearance = current_clearance(observation_frame, robot_state, occupancy)
        minimum_actual_clearance = min(minimum_actual_clearance, actual_clearance)
        human_state = estimator.update(observation_frame)
        prediction = predictor.predict(human_state, config.prediction.horizons_seconds)
        wall_times = tuple(item.timestamp for item in prediction.slices)
        trajectory_times = tuple(
            simulation.trajectory_time + item.horizon_seconds for item in prediction.slices
        )
        robot_trajectory = simulation.planned_link_trajectory(wall_times, trajectory_times)
        predicted_risk = (
            risk_engine.assess(prediction, robot_trajectory)
            if truth_frame.timestamp >= 0.5
            else None
        )
        decision = controller.update(
            truth_frame.timestamp,
            current_clearance_m=measured_clearance,
            predicted_risk=predicted_risk,
        )
        if previous_mode is SafetyMode.NORMAL and decision.mode is not SafetyMode.NORMAL:
            interventions += 1
            if first_intervention_timestamp is None:
                first_intervention_timestamp = truth_frame.timestamp
        if previous_mode is not SafetyMode.CRITICAL and decision.mode is SafetyMode.CRITICAL:
            stops += 1
            if first_stop_timestamp is None:
                first_stop_timestamp = truth_frame.timestamp
        previous_mode = decision.mode
        velocity_scale = decision.command.velocity_scale

    return ControllerMetrics(
        controller=name,
        minimum_actual_clearance_m=minimum_actual_clearance,
        intervention_count=interventions,
        stop_count=stops,
        idle_time_seconds=idle_time,
        final_task_progress_seconds=simulation.trajectory_time,
        final_mode=previous_mode,
        first_intervention_timestamp=first_intervention_timestamp,
        first_stop_timestamp=first_stop_timestamp,
    )
