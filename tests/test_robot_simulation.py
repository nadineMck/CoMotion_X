from pathlib import Path

import numpy as np

from comotion_x.estimation.state_estimator import HumanStateEstimator
from comotion_x.human_model.scenarios import generate_scenario
from comotion_x.prediction.motion_predictor import HumanMotionPredictor
from comotion_x.robot.simulation import ARM_JOINT_NAMES, PandaSimulation

MODEL_PATH = Path("third_party/mujoco_menagerie/franka_emika_panda/comotion_scene.xml")


def make_simulation() -> PandaSimulation:
    return PandaSimulation(
        model_path=MODEL_PATH,
        control_timestep_seconds=0.01,
        move_duration_seconds=0.5,
    )


def test_panda_model_and_state_load() -> None:
    simulation = make_simulation()
    state = simulation.state()

    assert simulation.model.nq == 9
    assert len(state.joint_positions) == len(ARM_JOINT_NAMES) == 7
    assert set(state.link_positions_m) == {
        "link0",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "hand",
    }
    assert len(simulation.human_marker_names) == 9


def test_planned_trajectory_is_time_aligned() -> None:
    simulation = make_simulation()
    trajectory = simulation.planned_trajectory(0.0, 0.5, 0.1)

    assert len(trajectory.times) == 6
    assert trajectory.times[0] == 0.0
    assert trajectory.times[-1] == 0.5
    assert all(len(configuration) == 7 for configuration in trajectory.joint_positions)
    assert not np.allclose(trajectory.joint_positions[0], trajectory.joint_positions[-1])


def test_planned_link_trajectory_has_world_positions() -> None:
    simulation = make_simulation()
    trajectory = simulation.planned_link_trajectory((0.1, 0.3, 0.5))

    assert [item.timestamp for item in trajectory.slices] == [0.1, 0.3, 0.5]
    expected_links = set(simulation.state().link_positions_m)
    assert all(set(item.link_positions_m) == expected_links for item in trajectory.slices)


def test_reaching_run_moves_end_effector() -> None:
    simulation = make_simulation()
    summary = simulation.run(duration_seconds=1.1)

    assert summary.physics_steps == 110
    assert summary.completed_moves >= 2
    assert summary.end_effector_path_length_m > 0.05
    assert np.isfinite(summary.final_joint_error_rad)


def test_zero_velocity_scale_freezes_task_clock() -> None:
    simulation = make_simulation()

    simulation.step(velocity_scale=0.0)

    assert simulation.data.time > 0
    assert simulation.trajectory_time == 0.0


def test_human_pose_updates_world_frame_markers() -> None:
    simulation = make_simulation()
    frame = generate_scenario("crossing").observation_frames[0]

    simulation.set_human_pose(frame)

    mocap_id = simulation._human_mocap_ids["right_wrist"]
    assert np.allclose(simulation.data.mocap_pos[mocap_id], frame.joints["right_wrist"].position_m)


def test_human_prediction_updates_future_wrist_markers() -> None:
    simulation = make_simulation()
    scenario = generate_scenario("crossing", duration_seconds=1.0, frames_per_second=20.0)
    estimator = HumanStateEstimator()
    state = None
    for frame in scenario.observation_frames[:5]:
        state = estimator.update(frame)
    assert state is not None
    prediction = HumanMotionPredictor().predict(state, (0.1, 0.2, 0.3, 0.5))

    simulation.set_human_prediction(prediction)

    predicted_wrist = prediction.slices[0].joints["right_wrist"]
    mocap_id = simulation._prediction_mocap_ids[100]
    geom_id = simulation._prediction_geom_ids[100]
    assert np.allclose(simulation.data.mocap_pos[mocap_id], predicted_wrist.mean_position_m)
    assert simulation.model.geom_size[geom_id, 0] >= 0.025
    assert simulation.model.geom_rgba[geom_id, 3] > 0
