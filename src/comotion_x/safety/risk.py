"""Time-aligned deterministic human-robot clearance assessment."""

from __future__ import annotations

from dataclasses import dataclass

from comotion_x.prediction.motion_predictor import HumanPrediction
from comotion_x.robot.simulation import RobotLinkTrajectory
from comotion_x.safety.geometry import primitive_distance
from comotion_x.safety.occupancy import OccupancyParameters, human_occupancy, robot_occupancy


@dataclass(frozen=True, slots=True)
class RiskSlice:
    horizon_seconds: float
    timestamp: float
    clearance_m: float
    human_primitive: str
    robot_primitive: str


@dataclass(frozen=True, slots=True)
class CollisionRisk:
    minimum_clearance_m: float
    time_to_closest_seconds: float
    collision_predicted: bool
    slices: tuple[RiskSlice, ...]


class CollisionRiskEngine:
    def __init__(self, parameters: OccupancyParameters | None = None) -> None:
        self.parameters = parameters or OccupancyParameters()

    def assess(
        self, human_prediction: HumanPrediction, robot_trajectory: RobotLinkTrajectory
    ) -> CollisionRisk:
        if len(human_prediction.slices) != len(robot_trajectory.slices):
            raise ValueError("human and robot trajectories must have equal lengths")
        risk_slices: list[RiskSlice] = []
        for human_slice, robot_slice in zip(
            human_prediction.slices, robot_trajectory.slices, strict=True
        ):
            if abs(human_slice.timestamp - robot_slice.timestamp) > 1e-8:
                raise ValueError("human and robot trajectory timestamps are not aligned")
            human_primitives = human_occupancy(human_slice, self.parameters)
            robot_primitives = robot_occupancy(robot_slice, self.parameters)
            if not human_primitives or not robot_primitives:
                raise ValueError("occupancy cannot be empty")
            closest = min(
                (
                    primitive_distance(human_primitive, robot_primitive)
                    for human_primitive in human_primitives
                    for robot_primitive in robot_primitives
                ),
                key=lambda distance: distance.clearance_m,
            )
            risk_slices.append(
                RiskSlice(
                    horizon_seconds=human_slice.horizon_seconds,
                    timestamp=human_slice.timestamp,
                    clearance_m=closest.clearance_m,
                    human_primitive=closest.first_name,
                    robot_primitive=closest.second_name,
                )
            )
        closest_slice = min(risk_slices, key=lambda item: item.clearance_m)
        return CollisionRisk(
            minimum_clearance_m=closest_slice.clearance_m,
            time_to_closest_seconds=closest_slice.horizon_seconds,
            collision_predicted=closest_slice.clearance_m <= 0,
            slices=tuple(risk_slices),
        )

