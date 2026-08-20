"""Dependency-light 3D sphere and capsule distance calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Sphere:
    name: str
    center_m: tuple[float, float, float]
    radius_m: float

    def __post_init__(self) -> None:
        _validate_point(self.center_m)
        if self.radius_m <= 0:
            raise ValueError("sphere radius must be positive")


@dataclass(frozen=True, slots=True)
class Capsule:
    name: str
    start_m: tuple[float, float, float]
    end_m: tuple[float, float, float]
    radius_m: float

    def __post_init__(self) -> None:
        _validate_point(self.start_m)
        _validate_point(self.end_m)
        if self.radius_m <= 0:
            raise ValueError("capsule radius must be positive")


Primitive = Sphere | Capsule


@dataclass(frozen=True, slots=True)
class PrimitiveDistance:
    first_name: str
    second_name: str
    centerline_distance_m: float
    clearance_m: float

    @property
    def overlaps(self) -> bool:
        return self.clearance_m <= 0


def primitive_distance(first: Primitive, second: Primitive) -> PrimitiveDistance:
    first_start, first_end, first_radius = _segment(first)
    second_start, second_end, second_radius = _segment(second)
    centerline = segment_distance(first_start, first_end, second_start, second_end)
    return PrimitiveDistance(
        first_name=first.name,
        second_name=second.name,
        centerline_distance_m=centerline,
        clearance_m=centerline - first_radius - second_radius,
    )


def segment_distance(
    first_start: tuple[float, float, float] | FloatArray,
    first_end: tuple[float, float, float] | FloatArray,
    second_start: tuple[float, float, float] | FloatArray,
    second_end: tuple[float, float, float] | FloatArray,
) -> float:
    """Return the shortest Euclidean distance between two finite 3D segments."""
    p1 = np.asarray(first_start, dtype=float)
    q1 = np.asarray(first_end, dtype=float)
    p2 = np.asarray(second_start, dtype=float)
    q2 = np.asarray(second_end, dtype=float)
    direction1 = q1 - p1
    direction2 = q2 - p2
    offset = p1 - p2
    a = float(direction1 @ direction1)
    e = float(direction2 @ direction2)
    epsilon = 1e-12

    if a <= epsilon and e <= epsilon:
        return float(np.linalg.norm(p1 - p2))
    if a <= epsilon:
        first_parameter = 0.0
        second_parameter = float(np.clip((direction2 @ offset) / e, 0.0, 1.0))
    else:
        c = float(direction1 @ offset)
        if e <= epsilon:
            second_parameter = 0.0
            first_parameter = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(direction1 @ direction2)
            denominator = a * e - b * b
            first_parameter = (
                float(np.clip((b * float(direction2 @ offset) - c * e) / denominator, 0, 1))
                if abs(denominator) > epsilon
                else 0.0
            )
            second_parameter = (b * first_parameter + float(direction2 @ offset)) / e
            if second_parameter < 0.0:
                second_parameter = 0.0
                first_parameter = float(np.clip(-c / a, 0.0, 1.0))
            elif second_parameter > 1.0:
                second_parameter = 1.0
                first_parameter = float(np.clip((b - c) / a, 0.0, 1.0))

    first_closest = p1 + direction1 * first_parameter
    second_closest = p2 + direction2 * second_parameter
    return float(np.linalg.norm(first_closest - second_closest))


def _segment(primitive: Primitive) -> tuple[FloatArray, FloatArray, float]:
    if isinstance(primitive, Sphere):
        center = np.asarray(primitive.center_m, dtype=float)
        return center, center, primitive.radius_m
    return (
        np.asarray(primitive.start_m, dtype=float),
        np.asarray(primitive.end_m, dtype=float),
        primitive.radius_m,
    )


def _validate_point(point: tuple[float, float, float]) -> None:
    values = np.asarray(point, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("geometry point must contain three finite values")

