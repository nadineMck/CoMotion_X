"""Typed data contracts shared across perception, prediction, safety, and control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


class SafetyMode(StrEnum):
    NORMAL = "normal"
    CAUTION = "caution"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class JointObservation:
    position_m: Vector3
    confidence: float

    def __post_init__(self) -> None:
        if len(self.position_m) != 3 or not all(math.isfinite(v) for v in self.position_m):
            raise ValueError("joint position must contain three finite values")
        if not 0 <= self.confidence <= 1:
            raise ValueError("joint confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class PoseFrame:
    timestamp: float
    frame_id: str
    joints: dict[str, JointObservation]

    def __post_init__(self) -> None:
        if self.timestamp < 0 or not math.isfinite(self.timestamp):
            raise ValueError("pose timestamp must be finite and non-negative")
        if not self.frame_id:
            raise ValueError("pose frame_id must not be empty")


@dataclass(frozen=True, slots=True)
class JointState:
    position_m: Vector3
    velocity_mps: Vector3
    covariance: Matrix3


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    timestamp: float
    min_distance_m: float
    time_to_closest_seconds: float | None
    collision_probability: float
    mode: SafetyMode

    def __post_init__(self) -> None:
        if self.timestamp < 0 or not math.isfinite(self.timestamp):
            raise ValueError("risk timestamp must be finite and non-negative")
        if math.isnan(self.min_distance_m) or self.min_distance_m < 0:
            raise ValueError("minimum distance must be non-negative")
        if self.time_to_closest_seconds is not None and self.time_to_closest_seconds < 0:
            raise ValueError("time to closest approach must be non-negative")
        if not 0 <= self.collision_probability <= 1:
            raise ValueError("collision probability must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ControlCommand:
    velocity_scale: float
    stop: bool = False
    replan_requested: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.velocity_scale <= 1:
            raise ValueError("velocity scale must be between 0 and 1")
        if self.stop and self.velocity_scale != 0:
            raise ValueError("a stop command must use zero velocity scale")
