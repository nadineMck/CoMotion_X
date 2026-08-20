"""Evaluate wrist prediction error and uncertainty coverage by horizon."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from comotion_x.estimation.state_estimator import HumanStateEstimator
from comotion_x.human_model.scenarios import HumanScenario
from comotion_x.prediction.motion_predictor import HumanMotionPredictor

CHI_SQUARE_3D_95 = 7.814727903


@dataclass(frozen=True, slots=True)
class HorizonMetrics:
    horizon_seconds: float
    samples: int
    mean_error_m: float
    root_mean_square_error_m: float
    coverage_95: float
    mean_uncertainty_radius_m: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "samples": self.samples,
            "mean_error_m": round(self.mean_error_m, 6),
            "root_mean_square_error_m": round(self.root_mean_square_error_m, 6),
            "coverage_95": round(self.coverage_95, 6),
            "mean_uncertainty_radius_m": round(self.mean_uncertainty_radius_m, 6),
        }


def evaluate_scenario_prediction(
    scenario: HumanScenario,
    horizons_seconds: tuple[float, ...],
    *,
    observation_std_m: float,
    acceleration_std_mps2: float,
    initial_velocity_std_mps: float,
    joint_name: str = "right_wrist",
) -> tuple[HorizonMetrics, ...]:
    estimator = HumanStateEstimator(
        observation_std_m=observation_std_m,
        acceleration_std_mps2=acceleration_std_mps2,
        initial_velocity_std_mps=initial_velocity_std_mps,
    )
    predictor = HumanMotionPredictor(acceleration_std_mps2=acceleration_std_mps2)
    sample_period = 1.0 / scenario.frames_per_second
    errors: dict[float, list[float]] = {horizon: [] for horizon in horizons_seconds}
    covered: dict[float, list[bool]] = {horizon: [] for horizon in horizons_seconds}
    radii: dict[float, list[float]] = {horizon: [] for horizon in horizons_seconds}

    for frame_index, observation_frame in enumerate(scenario.observation_frames):
        state = estimator.update(observation_frame)
        if joint_name not in state.joints or frame_index < 2:
            continue
        prediction = predictor.predict(state, horizons_seconds)
        for prediction_slice in prediction.slices:
            target_index = frame_index + round(prediction_slice.horizon_seconds / sample_period)
            if target_index >= len(scenario.ground_truth_frames):
                continue
            predicted_joint = prediction_slice.joints[joint_name]
            truth = np.asarray(
                scenario.ground_truth_frames[target_index].joints[joint_name].position_m
            )
            mean = np.asarray(predicted_joint.mean_position_m)
            covariance = np.asarray(predicted_joint.covariance)
            residual = truth - mean
            error = float(np.linalg.norm(residual))
            mahalanobis_squared = float(residual @ np.linalg.pinv(covariance) @ residual)
            largest_variance = float(np.linalg.eigvalsh(covariance).max())
            errors[prediction_slice.horizon_seconds].append(error)
            covered[prediction_slice.horizon_seconds].append(
                mahalanobis_squared <= CHI_SQUARE_3D_95
            )
            radii[prediction_slice.horizon_seconds].append(
                math_sqrt(CHI_SQUARE_3D_95 * largest_variance)
            )

    metrics: list[HorizonMetrics] = []
    for horizon in horizons_seconds:
        horizon_errors = np.asarray(errors[horizon])
        if not len(horizon_errors):
            raise ValueError(f"scenario is too short to evaluate horizon {horizon}")
        metrics.append(
            HorizonMetrics(
                horizon_seconds=horizon,
                samples=len(horizon_errors),
                mean_error_m=float(horizon_errors.mean()),
                root_mean_square_error_m=float(np.sqrt(np.mean(horizon_errors**2))),
                coverage_95=float(np.mean(covered[horizon])),
                mean_uncertainty_radius_m=float(np.mean(radii[horizon])),
            )
        )
    return tuple(metrics)


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))

