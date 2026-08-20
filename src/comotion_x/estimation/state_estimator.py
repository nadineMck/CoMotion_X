"""Per-joint temporal state estimation with dropout handling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from comotion_x.core.models import PoseFrame, Vector3
from comotion_x.estimation.kalman_filter import ConstantVelocityKalmanFilter

Matrix6 = tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class JointEstimate:
    position_m: Vector3
    velocity_mps: Vector3
    covariance: Matrix6
    missed_frames: int

    @property
    def position_covariance(self) -> tuple[Vector3, Vector3, Vector3]:
        return tuple(row[:3] for row in self.covariance[:3])  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class HumanStateFrame:
    timestamp: float
    frame_id: str
    joints: dict[str, JointEstimate]


class HumanStateEstimator:
    def __init__(
        self,
        *,
        observation_std_m: float = 0.015,
        acceleration_std_mps2: float = 1.5,
        initial_velocity_std_mps: float = 0.75,
    ) -> None:
        self.observation_std_m = observation_std_m
        self.acceleration_std_mps2 = acceleration_std_mps2
        self.initial_velocity_std_mps = initial_velocity_std_mps
        self._filters: dict[str, ConstantVelocityKalmanFilter] = {}
        self._missed_frames: dict[str, int] = {}
        self._last_timestamp: float | None = None
        self._frame_id: str | None = None

    def update(self, frame: PoseFrame) -> HumanStateFrame:
        if self._frame_id is not None and frame.frame_id != self._frame_id:
            raise ValueError("state estimator cannot mix coordinate frames")
        if self._last_timestamp is not None and frame.timestamp <= self._last_timestamp:
            raise ValueError("pose timestamps must be strictly increasing")
        self._frame_id = frame.frame_id

        if self._last_timestamp is not None:
            delta_time = frame.timestamp - self._last_timestamp
            for joint_filter in self._filters.values():
                joint_filter.predict(delta_time)
        self._last_timestamp = frame.timestamp

        for joint_name in self._filters:
            self._missed_frames[joint_name] += 1
        for joint_name, observation in frame.joints.items():
            if joint_name not in self._filters:
                self._filters[joint_name] = ConstantVelocityKalmanFilter(
                    np.asarray(observation.position_m),
                    observation_std_m=self.observation_std_m,
                    acceleration_std_mps2=self.acceleration_std_mps2,
                    initial_velocity_std_mps=self.initial_velocity_std_mps,
                )
            else:
                self._filters[joint_name].update(
                    np.asarray(observation.position_m), observation.confidence
                )
            self._missed_frames[joint_name] = 0

        return HumanStateFrame(
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            joints={
                name: _estimate(joint_filter, self._missed_frames[name])
                for name, joint_filter in self._filters.items()
            },
        )


def _estimate(
    joint_filter: ConstantVelocityKalmanFilter, missed_frames: int
) -> JointEstimate:
    state = joint_filter.state
    covariance = joint_filter.covariance
    return JointEstimate(
        position_m=tuple(float(value) for value in state[:3]),  # type: ignore[arg-type]
        velocity_mps=tuple(float(value) for value in state[3:]),  # type: ignore[arg-type]
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        missed_frames=missed_frames,
    )

