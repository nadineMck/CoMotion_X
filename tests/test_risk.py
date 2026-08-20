import numpy as np
import pytest

from comotion_x.prediction.motion_predictor import (
    HumanPrediction,
    PredictedJoint,
    PredictionSlice,
)
from comotion_x.robot.simulation import RobotLinkTrajectory, RobotLinkTrajectorySlice
from comotion_x.safety.occupancy import OccupancyParameters, human_occupancy
from comotion_x.safety.risk import CollisionRiskEngine


def predicted_wrist(
    position: tuple[float, float, float], variance: float = 0.0
) -> PredictedJoint:
    covariance = (
        (variance, 0.0, 0.0),
        (0.0, variance, 0.0),
        (0.0, 0.0, variance),
    )
    return PredictedJoint(mean_position_m=position, covariance=covariance)


def human_prediction(position: tuple[float, float, float]) -> HumanPrediction:
    return HumanPrediction(
        source_timestamp=0.0,
        frame_id="world",
        slices=(
            PredictionSlice(
                horizon_seconds=0.1,
                timestamp=0.1,
                joints={"right_wrist": predicted_wrist(position)},
            ),
        ),
    )


def robot_trajectory(position: tuple[float, float, float]) -> RobotLinkTrajectory:
    links = {f"link{index}": position for index in range(8)}
    links["hand"] = position
    return RobotLinkTrajectory(
        slices=(RobotLinkTrajectorySlice(timestamp=0.1, link_positions_m=links),)
    )


def test_risk_engine_orders_safe_and_collision_cases() -> None:
    engine = CollisionRiskEngine(OccupancyParameters(uncertainty_sigma=0.0))

    safe = engine.assess(human_prediction((1.0, 0.0, 0.0)), robot_trajectory((0.0, 0.0, 0.0)))
    collision = engine.assess(
        human_prediction((0.05, 0.0, 0.0)), robot_trajectory((0.0, 0.0, 0.0))
    )

    assert safe.minimum_clearance_m > 0
    assert not safe.collision_predicted
    assert collision.minimum_clearance_m < 0
    assert collision.collision_predicted
    assert collision.time_to_closest_seconds == 0.1


def test_uncertainty_inflates_human_occupancy() -> None:
    prediction_slice = PredictionSlice(
        horizon_seconds=0.5,
        timestamp=0.5,
        joints={"right_wrist": predicted_wrist((0.0, 0.0, 0.0), variance=0.01)},
    )

    deterministic = human_occupancy(
        prediction_slice, OccupancyParameters(uncertainty_sigma=0.0)
    )[0]
    uncertain = human_occupancy(
        prediction_slice, OccupancyParameters(uncertainty_sigma=2.0)
    )[0]

    assert uncertain.radius_m == pytest.approx(deterministic.radius_m + 0.2)


def test_risk_rejects_misaligned_timestamps() -> None:
    prediction = human_prediction((1.0, 0.0, 0.0))
    trajectory = robot_trajectory((0.0, 0.0, 0.0))
    wrong_slice = RobotLinkTrajectorySlice(
        timestamp=0.2,
        link_positions_m=trajectory.slices[0].link_positions_m,
    )

    with pytest.raises(ValueError, match="not aligned"):
        CollisionRiskEngine().assess(
            prediction, RobotLinkTrajectory(slices=(wrong_slice,))
        )


def test_geometry_covariance_is_positive_semidefinite() -> None:
    joint = predicted_wrist((0.0, 0.0, 0.0), variance=0.01)
    assert np.linalg.eigvalsh(joint.covariance).min() >= 0
