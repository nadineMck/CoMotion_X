import numpy as np
import pytest

from comotion_x.estimation.state_estimator import HumanStateEstimator
from comotion_x.evaluation.prediction import evaluate_scenario_prediction
from comotion_x.human_model.scenarios import generate_scenario
from comotion_x.prediction.motion_predictor import HumanMotionPredictor


def test_prediction_covariance_grows_with_horizon() -> None:
    scenario = generate_scenario("crossing", duration_seconds=1.0, frames_per_second=20.0)
    estimator = HumanStateEstimator()
    state = None
    for observation in scenario.observation_frames[:5]:
        state = estimator.update(observation)
    assert state is not None

    prediction = HumanMotionPredictor().predict(state, (0.1, 0.3, 0.5))
    traces = [
        np.trace(prediction_slice.joints["right_wrist"].covariance)
        for prediction_slice in prediction.slices
    ]

    assert traces[0] < traces[1] < traces[2]
    assert [item.timestamp for item in prediction.slices] == pytest.approx([0.3, 0.5, 0.7])


def test_constant_velocity_scenario_has_finite_metrics() -> None:
    scenario = generate_scenario(
        "crossing",
        duration_seconds=2.0,
        frames_per_second=30.0,
        noise_standard_deviation_m=0.005,
    )

    metrics = evaluate_scenario_prediction(
        scenario,
        (0.1, 0.3, 0.5),
        observation_std_m=0.01,
        acceleration_std_mps2=1.0,
        initial_velocity_std_mps=0.75,
    )

    assert len(metrics) == 3
    assert all(metric.samples > 0 for metric in metrics)
    assert all(np.isfinite(metric.root_mean_square_error_m) for metric in metrics)
    assert metrics[-1].mean_uncertainty_radius_m > metrics[0].mean_uncertainty_radius_m
