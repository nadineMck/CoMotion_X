"""Constant-velocity Kalman filter for one 3D human joint."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def transition_matrix(delta_time: float) -> FloatArray:
    if delta_time < 0:
        raise ValueError("delta time must be non-negative")
    transition = np.eye(6, dtype=float)
    transition[:3, 3:] = np.eye(3) * delta_time
    return transition


def process_covariance(delta_time: float, acceleration_std: float) -> FloatArray:
    if delta_time < 0 or acceleration_std <= 0:
        raise ValueError("delta time must be non-negative and acceleration noise positive")
    variance = acceleration_std**2
    covariance = np.zeros((6, 6), dtype=float)
    covariance[:3, :3] = np.eye(3) * (delta_time**4 / 4.0) * variance
    covariance[:3, 3:] = np.eye(3) * (delta_time**3 / 2.0) * variance
    covariance[3:, :3] = covariance[:3, 3:]
    covariance[3:, 3:] = np.eye(3) * (delta_time**2) * variance
    return covariance


class ConstantVelocityKalmanFilter:
    def __init__(
        self,
        position_m: FloatArray,
        *,
        observation_std_m: float,
        acceleration_std_mps2: float,
        initial_velocity_std_mps: float,
    ) -> None:
        position = np.asarray(position_m, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("initial position must contain three finite values")
        if observation_std_m <= 0 or initial_velocity_std_mps <= 0:
            raise ValueError("filter uncertainty values must be positive")
        self.state = np.concatenate((position, np.zeros(3, dtype=float)))
        self.covariance = np.diag(
            [observation_std_m**2] * 3 + [initial_velocity_std_mps**2] * 3
        )
        self.observation_std_m = observation_std_m
        self.acceleration_std_mps2 = acceleration_std_mps2

    def predict(self, delta_time: float) -> None:
        transition = transition_matrix(delta_time)
        process_noise = process_covariance(delta_time, self.acceleration_std_mps2)
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        self.covariance = (self.covariance + self.covariance.T) * 0.5

    def update(self, position_m: FloatArray, confidence: float = 1.0) -> None:
        observation = np.asarray(position_m, dtype=float)
        if observation.shape != (3,) or not np.all(np.isfinite(observation)):
            raise ValueError("observation must contain three finite values")
        if not 0 < confidence <= 1:
            raise ValueError("observation confidence must be in (0, 1]")
        observation_model = np.zeros((3, 6), dtype=float)
        observation_model[:, :3] = np.eye(3)
        observation_variance = (self.observation_std_m / confidence) ** 2
        measurement_noise = np.eye(3) * observation_variance
        innovation = observation - observation_model @ self.state
        innovation_covariance = (
            observation_model @ self.covariance @ observation_model.T + measurement_noise
        )
        gain = self.covariance @ observation_model.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + gain @ innovation
        identity = np.eye(6)
        correction = identity - gain @ observation_model
        self.covariance = (
            correction @ self.covariance @ correction.T + gain @ measurement_noise @ gain.T
        )
        self.covariance = (self.covariance + self.covariance.T) * 0.5

