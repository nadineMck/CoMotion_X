import pytest

from comotion_x.core.models import ControlCommand, JointObservation, SafetyMode


def test_joint_observation_accepts_valid_input() -> None:
    joint = JointObservation(position_m=(0.1, 0.2, 0.3), confidence=0.9)

    assert joint.position_m == (0.1, 0.2, 0.3)


def test_joint_observation_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        JointObservation(position_m=(0.1, 0.2, 0.3), confidence=1.1)


def test_stop_command_requires_zero_velocity() -> None:
    with pytest.raises(ValueError, match="zero velocity"):
        ControlCommand(velocity_scale=0.2, stop=True)


def test_safety_mode_values_are_stable() -> None:
    assert SafetyMode.HIGH_RISK.value == "high_risk"

