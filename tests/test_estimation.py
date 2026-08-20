import numpy as np
import pytest

from comotion_x.core.models import JointObservation, PoseFrame
from comotion_x.estimation.state_estimator import HumanStateEstimator


def frame(timestamp: float, position: tuple[float, float, float] | None) -> PoseFrame:
    joints = (
        {"right_wrist": JointObservation(position_m=position, confidence=1.0)}
        if position is not None
        else {}
    )
    return PoseFrame(timestamp=timestamp, frame_id="world", joints=joints)


def test_filter_estimates_constant_velocity() -> None:
    estimator = HumanStateEstimator(
        observation_std_m=0.01,
        acceleration_std_mps2=0.2,
        initial_velocity_std_mps=1.0,
    )
    state = None
    for index in range(31):
        timestamp = index * 0.05
        position = (0.3 + 0.4 * timestamp, -0.2 * timestamp, 1.0)
        state = estimator.update(frame(timestamp, position))

    assert state is not None
    wrist = state.joints["right_wrist"]
    assert wrist.velocity_mps == pytest.approx((0.4, -0.2, 0.0), abs=0.025)
    assert wrist.position_m == pytest.approx((0.9, -0.3, 1.0), abs=0.01)


def test_missing_observation_propagates_state_and_uncertainty() -> None:
    estimator = HumanStateEstimator(
        observation_std_m=0.01,
        acceleration_std_mps2=0.5,
        initial_velocity_std_mps=1.0,
    )
    estimator.update(frame(0.0, (0.0, 0.0, 0.0)))
    observed = estimator.update(frame(0.1, (0.1, 0.0, 0.0)))
    missing = estimator.update(frame(0.2, None))

    assert missing.joints["right_wrist"].missed_frames == 1
    assert (
        missing.joints["right_wrist"].position_m[0]
        > observed.joints["right_wrist"].position_m[0]
    )
    observed_trace = np.trace(observed.joints["right_wrist"].position_covariance)
    missing_trace = np.trace(missing.joints["right_wrist"].position_covariance)
    assert missing_trace > observed_trace


def test_estimator_rejects_non_increasing_timestamps() -> None:
    estimator = HumanStateEstimator()
    estimator.update(frame(0.1, (0.0, 0.0, 0.0)))

    with pytest.raises(ValueError, match="strictly increasing"):
        estimator.update(frame(0.1, (0.0, 0.0, 0.0)))
