"""Convert predicted humans and planned robot links into simple occupied volumes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from comotion_x.core.models import PoseFrame
from comotion_x.prediction.motion_predictor import PredictedJoint, PredictionSlice
from comotion_x.robot.simulation import RobotLinkTrajectorySlice, RobotState
from comotion_x.safety.geometry import Capsule, Primitive, Sphere, primitive_distance

ARM_SEGMENTS = (
    ("left_upper_arm", "left_shoulder", "left_elbow"),
    ("left_forearm", "left_elbow", "left_wrist"),
    ("right_upper_arm", "right_shoulder", "right_elbow"),
    ("right_forearm", "right_elbow", "right_wrist"),
)


@dataclass(frozen=True, slots=True)
class OccupancyParameters:
    human_wrist_radius_m: float = 0.07
    human_arm_radius_m: float = 0.06
    human_torso_radius_m: float = 0.18
    robot_link_radius_m: float = 0.055
    robot_hand_radius_m: float = 0.075
    uncertainty_sigma: float = 2.0


def human_occupancy(
    prediction_slice: PredictionSlice, parameters: OccupancyParameters
) -> tuple[Primitive, ...]:
    primitives: list[Primitive] = []
    joints = prediction_slice.joints
    for wrist_name in ("left_wrist", "right_wrist"):
        if wrist_name in joints:
            joint = joints[wrist_name]
            primitives.append(
                Sphere(
                    name=f"human_{wrist_name}",
                    center_m=joint.mean_position_m,
                    radius_m=(
                        parameters.human_wrist_radius_m
                        + _uncertainty_radius(joint.covariance, parameters.uncertainty_sigma)
                    ),
                )
            )
    if "torso" in joints:
        torso = joints["torso"]
        primitives.append(
            Sphere(
                name="human_torso",
                center_m=torso.mean_position_m,
                radius_m=(
                    parameters.human_torso_radius_m
                    + _uncertainty_radius(torso.covariance, parameters.uncertainty_sigma)
                ),
            )
        )
    for segment_name, start_name, end_name in ARM_SEGMENTS:
        if start_name not in joints or end_name not in joints:
            continue
        start = joints[start_name]
        end = joints[end_name]
        inflation = max(
            _uncertainty_radius(start.covariance, parameters.uncertainty_sigma),
            _uncertainty_radius(end.covariance, parameters.uncertainty_sigma),
        )
        primitives.append(
            Capsule(
                name=f"human_{segment_name}",
                start_m=start.mean_position_m,
                end_m=end.mean_position_m,
                radius_m=parameters.human_arm_radius_m + inflation,
            )
        )
    return tuple(primitives)


def robot_occupancy(
    trajectory_slice: RobotLinkTrajectorySlice, parameters: OccupancyParameters
) -> tuple[Primitive, ...]:
    links = trajectory_slice.link_positions_m
    primitives: list[Primitive] = []
    ordered = tuple(f"link{index}" for index in range(8)) + ("hand",)
    for start_name, end_name in zip(ordered, ordered[1:], strict=False):
        primitives.append(
            Capsule(
                name=f"robot_{start_name}_{end_name}",
                start_m=links[start_name],
                end_m=links[end_name],
                radius_m=parameters.robot_link_radius_m,
            )
        )
    primitives.append(
        Sphere(
            name="robot_hand",
            center_m=links["hand"],
            radius_m=parameters.robot_hand_radius_m,
        )
    )
    return tuple(primitives)


def current_clearance(
    human_frame: PoseFrame,
    robot_state: RobotState,
    parameters: OccupancyParameters,
) -> float:
    if human_frame.frame_id != "world":
        raise ValueError("current human occupancy must use the world frame")
    zero_covariance = ((0.0, 0.0, 0.0),) * 3
    human_slice = PredictionSlice(
        horizon_seconds=0.0,
        timestamp=human_frame.timestamp,
        joints={
            name: PredictedJoint(
                mean_position_m=observation.position_m,
                covariance=zero_covariance,
            )
            for name, observation in human_frame.joints.items()
        },
    )
    robot_slice = RobotLinkTrajectorySlice(
        timestamp=robot_state.timestamp,
        link_positions_m=robot_state.link_positions_m,
    )
    human_primitives = human_occupancy(
        human_slice,
        OccupancyParameters(
            human_wrist_radius_m=parameters.human_wrist_radius_m,
            human_arm_radius_m=parameters.human_arm_radius_m,
            human_torso_radius_m=parameters.human_torso_radius_m,
            robot_link_radius_m=parameters.robot_link_radius_m,
            robot_hand_radius_m=parameters.robot_hand_radius_m,
            uncertainty_sigma=0.0,
        ),
    )
    robot_primitives = robot_occupancy(robot_slice, parameters)
    return min(
        primitive_distance(human, robot).clearance_m
        for human in human_primitives
        for robot in robot_primitives
    )


def _uncertainty_radius(
    covariance: tuple[tuple[float, float, float], ...], sigma: float
) -> float:
    eigenvalues = np.linalg.eigvalsh(np.asarray(covariance, dtype=float))
    return sigma * float(np.sqrt(max(0.0, float(eigenvalues.max()))))
