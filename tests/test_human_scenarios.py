import json

import numpy as np
import pytest

from comotion_x.human_model.replay import HumanReplay
from comotion_x.human_model.scenarios import JOINT_NAMES, ScenarioName, generate_scenario


@pytest.mark.parametrize("scenario_name", list(ScenarioName))
def test_every_scenario_is_world_frame_and_complete(scenario_name: ScenarioName) -> None:
    scenario = generate_scenario(
        scenario_name,
        duration_seconds=2.0,
        frames_per_second=10.0,
        seed=7,
    )

    assert len(scenario.ground_truth_frames) == 21
    assert scenario.ground_truth_frames[0].timestamp == 0.0
    assert scenario.ground_truth_frames[-1].timestamp == 2.0
    assert all(frame.frame_id == "world" for frame in scenario.ground_truth_frames)
    assert all(set(frame.joints) == set(JOINT_NAMES) for frame in scenario.ground_truth_frames)


def test_scenarios_are_deterministic_for_fixed_seed() -> None:
    first = generate_scenario("crossing", seed=91, noise_standard_deviation_m=0.01)
    second = generate_scenario("crossing", seed=91, noise_standard_deviation_m=0.01)

    assert first.observation_frames == second.observation_frames


def test_crossing_and_stationary_have_expected_wrist_motion() -> None:
    crossing = generate_scenario("crossing")
    stationary = generate_scenario("stationary")

    crossing_start = crossing.ground_truth_frames[0].joints["right_wrist"].position_m
    crossing_end = crossing.ground_truth_frames[-1].joints["right_wrist"].position_m
    stationary_start = stationary.ground_truth_frames[0].joints["right_wrist"].position_m
    stationary_end = stationary.ground_truth_frames[-1].joints["right_wrist"].position_m

    assert crossing_end[1] - crossing_start[1] == pytest.approx(1.1)
    assert stationary_start == stationary_end


def test_occlusion_removes_only_observation_not_ground_truth() -> None:
    scenario = generate_scenario("occlusion", duration_seconds=2.0, frames_per_second=20.0)
    middle = scenario.observation_frames[23]

    assert "right_wrist" not in middle.joints
    assert "right_wrist" in scenario.ground_truth_frames[23].joints


def test_noise_and_dropout_change_observations_only() -> None:
    scenario = generate_scenario(
        "stationary",
        duration_seconds=1.0,
        frames_per_second=10.0,
        seed=5,
        noise_standard_deviation_m=0.02,
        dropout_probability=0.2,
    )

    assert all(len(frame.joints) == len(JOINT_NAMES) for frame in scenario.ground_truth_frames)
    assert any(len(frame.joints) < len(JOINT_NAMES) for frame in scenario.observation_frames)
    paired = [
        (truth.joints[name].position_m, observed.joints[name].position_m)
        for truth, observed in zip(
            scenario.ground_truth_frames, scenario.observation_frames, strict=True
        )
        for name in observed.joints
    ]
    assert any(not np.allclose(truth, observed) for truth, observed in paired)


def test_replay_holds_latest_available_frame() -> None:
    scenario = generate_scenario("crossing", duration_seconds=1.0, frames_per_second=10.0)
    replay = HumanReplay(scenario)

    assert replay.frame_at(0.19).timestamp == pytest.approx(0.1)
    assert replay.frame_at(99.0).timestamp == pytest.approx(1.0)


def test_scenario_exports_machine_readable_json(tmp_path) -> None:
    scenario = generate_scenario("near_miss", duration_seconds=1.0, frames_per_second=5.0)
    output = tmp_path / "near_miss.json"

    scenario.export_json(output)
    payload = json.loads(output.read_text())

    assert payload["schema_version"] == 1
    assert payload["frame_id"] == "world"
    assert len(payload["ground_truth_frames"]) == 6
