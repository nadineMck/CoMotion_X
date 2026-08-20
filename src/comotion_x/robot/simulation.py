"""MuJoCo Franka Panda simulation and joint-space reaching baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
LINK_BODY_NAMES = tuple(f"link{index}" for index in range(8)) + ("hand",)

HOME_CONFIGURATION = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
REACH_CONFIGURATION = np.array([0.45, -0.35, 0.25, -2.05, 0.20, 1.85, -0.35])


@dataclass(frozen=True, slots=True)
class RobotState:
    timestamp: float
    joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    link_positions_m: dict[str, tuple[float, float, float]]


@dataclass(frozen=True, slots=True)
class PlannedTrajectory:
    times: tuple[float, ...]
    joint_positions: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class SimulationSummary:
    duration_seconds: float
    physics_steps: int
    completed_moves: int
    end_effector_path_length_m: float
    final_joint_error_rad: float
    final_end_effector_position_m: tuple[float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": round(self.duration_seconds, 6),
            "physics_steps": self.physics_steps,
            "completed_moves": self.completed_moves,
            "end_effector_path_length_m": round(self.end_effector_path_length_m, 6),
            "final_joint_error_rad": round(self.final_joint_error_rad, 6),
            "final_end_effector_position_m": [
                round(value, 6) for value in self.final_end_effector_position_m
            ],
        }


class PandaSimulation:
    """Headless MuJoCo simulation with a smooth repeated reaching command."""

    def __init__(
        self,
        model_path: Path | str,
        control_timestep_seconds: float = 0.01,
        move_duration_seconds: float = 1.5,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model not found: {self.model_path}")
        if control_timestep_seconds <= 0 or move_duration_seconds <= 0:
            raise ValueError("simulation timing values must be positive")

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path.resolve()))
        self.data = mujoco.MjData(self.model)
        self.control_timestep_seconds = control_timestep_seconds
        self.move_duration_seconds = move_duration_seconds
        self._joint_ids = np.array(
            [self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES]
        )
        self._qpos_addresses = self.model.jnt_qposadr[self._joint_ids]
        self._dof_addresses = self.model.jnt_dofadr[self._joint_ids]
        self._actuator_ids = np.array(
            [
                self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{index}")
                for index in range(1, 8)
            ]
        )
        self._link_ids = {
            name: self._name_id(mujoco.mjtObj.mjOBJ_BODY, name) for name in LINK_BODY_NAMES
        }
        self._hand_id = self._link_ids["hand"]
        self.reset()

    def _name_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, object_type, name)
        if identifier < 0:
            raise ValueError(f"Required MuJoCo object is missing: {name}")
        return identifier

    def reset(self) -> RobotState:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.model.opt.timestep = min(self.control_timestep_seconds, self.model.opt.timestep)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self) -> RobotState:
        positions = tuple(float(self.data.qpos[address]) for address in self._qpos_addresses)
        velocities = tuple(float(self.data.qvel[address]) for address in self._dof_addresses)
        links = {
            name: tuple(float(value) for value in self.data.xpos[identifier])
            for name, identifier in self._link_ids.items()
        }
        return RobotState(
            timestamp=float(self.data.time),
            joint_positions=positions,
            joint_velocities=velocities,
            link_positions_m=links,
        )

    def desired_configuration(self, timestamp: float) -> NDArray[np.float64]:
        if timestamp < 0:
            raise ValueError("trajectory time must be non-negative")
        segment = int(timestamp / self.move_duration_seconds)
        phase = (timestamp % self.move_duration_seconds) / self.move_duration_seconds
        blend = 0.5 - 0.5 * math.cos(math.pi * phase)
        start, goal = (
            (HOME_CONFIGURATION, REACH_CONFIGURATION)
            if segment % 2 == 0
            else (REACH_CONFIGURATION, HOME_CONFIGURATION)
        )
        return start + blend * (goal - start)

    def planned_trajectory(
        self, start_time: float, horizon_seconds: float, sample_period_seconds: float
    ) -> PlannedTrajectory:
        if horizon_seconds < 0 or sample_period_seconds <= 0:
            raise ValueError("trajectory horizon must be non-negative and sample period positive")
        offsets = np.arange(
            0.0,
            horizon_seconds + sample_period_seconds * 0.5,
            sample_period_seconds,
        )
        times = tuple(float(start_time + offset) for offset in offsets)
        positions = tuple(
            tuple(float(value) for value in self.desired_configuration(timestamp))
            for timestamp in times
        )
        return PlannedTrajectory(times=times, joint_positions=positions)

    def step(self) -> RobotState:
        desired = self.desired_configuration(float(self.data.time))
        self.data.ctrl[self._actuator_ids] = desired
        target_time = self.data.time + self.control_timestep_seconds
        while self.data.time < target_time - 1e-12:
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def run(self, duration_seconds: float) -> SimulationSummary:
        if duration_seconds <= 0:
            raise ValueError("simulation duration must be positive")
        self.reset()
        previous_hand = self.data.xpos[self._hand_id].copy()
        path_length = 0.0
        steps = math.ceil(duration_seconds / self.control_timestep_seconds)

        for _ in range(steps):
            state = self.step()
            current_hand = np.asarray(state.link_positions_m["hand"])
            path_length += float(np.linalg.norm(current_hand - previous_hand))
            previous_hand = current_hand

        desired = self.desired_configuration(float(self.data.time))
        actual = self.data.qpos[self._qpos_addresses]
        hand = tuple(float(value) for value in self.data.xpos[self._hand_id])
        return SimulationSummary(
            duration_seconds=float(self.data.time),
            physics_steps=steps,
            completed_moves=int(self.data.time / self.move_duration_seconds),
            end_effector_path_length_m=path_length,
            final_joint_error_rad=float(np.linalg.norm(desired - actual)),
            final_end_effector_position_m=hand,
        )
