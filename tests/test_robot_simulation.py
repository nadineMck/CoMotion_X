from pathlib import Path

import numpy as np

from comotion_x.robot.simulation import ARM_JOINT_NAMES, PandaSimulation

MODEL_PATH = Path("third_party/mujoco_menagerie/franka_emika_panda/scene.xml")


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


def test_planned_trajectory_is_time_aligned() -> None:
    simulation = make_simulation()
    trajectory = simulation.planned_trajectory(0.0, 0.5, 0.1)

    assert len(trajectory.times) == 6
    assert trajectory.times[0] == 0.0
    assert trajectory.times[-1] == 0.5
    assert all(len(configuration) == 7 for configuration in trajectory.joint_positions)
    assert not np.allclose(trajectory.joint_positions[0], trajectory.joint_positions[-1])


def test_reaching_run_moves_end_effector() -> None:
    simulation = make_simulation()
    summary = simulation.run(duration_seconds=1.1)

    assert summary.physics_steps == 110
    assert summary.completed_moves >= 2
    assert summary.end_effector_path_length_m > 0.05
    assert np.isfinite(summary.final_joint_error_rad)

