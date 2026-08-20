import pytest

from comotion_x.core.config import load_config
from comotion_x.core.models import SafetyMode
from comotion_x.evaluation.controllers import compare_controllers
from comotion_x.human_model.scenarios import generate_scenario
from comotion_x.safety.controller import (
    ControllerParameters,
    NoAwarenessController,
    PredictiveSafetyController,
    ReactiveSafetyController,
)
from comotion_x.safety.risk import CollisionRisk


def risk(clearance: float) -> CollisionRisk:
    return CollisionRisk(
        minimum_clearance_m=clearance,
        time_to_closest_seconds=0.3,
        collision_predicted=clearance <= 0,
        slices=(),
    )


def test_no_awareness_always_runs_normally() -> None:
    decision = NoAwarenessController().update(
        1.0, current_clearance_m=-1.0, predicted_risk=risk(-1.0)
    )

    assert decision.mode is SafetyMode.NORMAL
    assert decision.command.velocity_scale == 1.0


@pytest.mark.parametrize(
    ("clearance", "mode", "scale"),
    [
        (0.4, SafetyMode.NORMAL, 1.0),
        (0.2, SafetyMode.CAUTION, 0.6),
        (0.05, SafetyMode.HIGH_RISK, 0.25),
        (-0.01, SafetyMode.CRITICAL, 0.0),
    ],
)
def test_reactive_modes_map_to_velocity_commands(
    clearance: float, mode: SafetyMode, scale: float
) -> None:
    controller = ReactiveSafetyController(ControllerParameters())

    decision = controller.update(1.0, current_clearance_m=clearance, predicted_risk=None)

    assert decision.mode is mode
    assert decision.command.velocity_scale == scale
    assert decision.command.stop is (mode is SafetyMode.CRITICAL)


def test_predictive_controller_uses_future_not_current_clearance() -> None:
    controller = PredictiveSafetyController(ControllerParameters())

    decision = controller.update(
        1.0,
        current_clearance_m=0.5,
        predicted_risk=risk(-0.02),
    )

    assert decision.mode is SafetyMode.CRITICAL
    assert decision.command.stop


def test_hysteresis_and_dwell_prevent_immediate_release() -> None:
    parameters = ControllerParameters(hysteresis_m=0.03, minimum_dwell_seconds=0.2)
    controller = ReactiveSafetyController(parameters)
    controller.update(1.0, current_clearance_m=-0.01, predicted_risk=None)

    during_dwell = controller.update(1.1, current_clearance_m=0.5, predicted_risk=None)
    after_dwell = controller.update(1.3, current_clearance_m=0.5, predicted_risk=None)

    assert during_dwell.mode is SafetyMode.CRITICAL
    assert after_dwell.mode is SafetyMode.HIGH_RISK


def test_predictive_controller_uses_current_clearance_during_warmup() -> None:
    controller = PredictiveSafetyController(ControllerParameters())

    decision = controller.update(0.1, current_clearance_m=0.4, predicted_risk=None)

    assert decision.mode is SafetyMode.NORMAL
    assert "warm-up" in decision.reason


def test_predictive_crossing_intervenes_before_reactive() -> None:
    config = load_config("config/default.toml")
    scenario = generate_scenario(
        "crossing",
        duration_seconds=config.human.duration_seconds,
        frames_per_second=config.human.frames_per_second,
        seed=config.project.seed,
    )

    metrics = {item.controller: item for item in compare_controllers(scenario, config)}

    assert metrics["predictive"].first_intervention_timestamp is not None
    assert metrics["reactive"].first_intervention_timestamp is not None
    assert (
        metrics["predictive"].first_intervention_timestamp
        < metrics["reactive"].first_intervention_timestamp
    )
    assert (
        metrics["predictive"].minimum_actual_clearance_m
        > metrics["reactive"].minimum_actual_clearance_m
    )
