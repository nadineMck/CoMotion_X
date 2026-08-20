"""Uncertainty-aware constant-velocity prediction from filtered human state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from comotion_x.core.models import Vector3
from comotion_x.estimation.kalman_filter import process_covariance, transition_matrix
from comotion_x.estimation.state_estimator import HumanStateFrame


@dataclass(frozen=True, slots=True)
class PredictedJoint:
    mean_position_m: Vector3
    covariance: tuple[Vector3, Vector3, Vector3]


@dataclass(frozen=True, slots=True)
class PredictionSlice:
    horizon_seconds: float
    timestamp: float
    joints: dict[str, PredictedJoint]


@dataclass(frozen=True, slots=True)
class HumanPrediction:
    source_timestamp: float
    frame_id: str
    slices: tuple[PredictionSlice, ...]


class HumanMotionPredictor:
    def __init__(self, *, acceleration_std_mps2: float = 1.5) -> None:
        if acceleration_std_mps2 <= 0:
            raise ValueError("prediction acceleration noise must be positive")
        self.acceleration_std_mps2 = acceleration_std_mps2

    def predict(
        self, state_frame: HumanStateFrame, horizons_seconds: tuple[float, ...]
    ) -> HumanPrediction:
        if not horizons_seconds or any(horizon <= 0 for horizon in horizons_seconds):
            raise ValueError("prediction horizons must be non-empty and positive")
        if tuple(sorted(horizons_seconds)) != horizons_seconds:
            raise ValueError("prediction horizons must be in ascending order")

        slices: list[PredictionSlice] = []
        for horizon in horizons_seconds:
            transition = transition_matrix(horizon)
            process_noise = process_covariance(horizon, self.acceleration_std_mps2)
            joints: dict[str, PredictedJoint] = {}
            for name, estimate in state_frame.joints.items():
                state = np.asarray((*estimate.position_m, *estimate.velocity_mps))
                covariance = np.asarray(estimate.covariance)
                future_state = transition @ state
                future_covariance = transition @ covariance @ transition.T + process_noise
                position_covariance = future_covariance[:3, :3]
                joints[name] = PredictedJoint(
                    mean_position_m=tuple(  # type: ignore[arg-type]
                        float(value) for value in future_state[:3]
                    ),
                    covariance=tuple(  # type: ignore[arg-type]
                        tuple(float(value) for value in row) for row in position_covariance
                    ),
                )
            slices.append(
                PredictionSlice(
                    horizon_seconds=horizon,
                    timestamp=state_frame.timestamp + horizon,
                    joints=joints,
                )
            )
        return HumanPrediction(
            source_timestamp=state_frame.timestamp,
            frame_id=state_frame.frame_id,
            slices=tuple(slices),
        )

