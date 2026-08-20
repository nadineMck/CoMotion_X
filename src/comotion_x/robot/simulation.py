"""MuJoCo Franka Panda simulation and joint-space reaching baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from comotion_x.core.models import PoseFrame
from comotion_x.prediction.motion_predictor import HumanPrediction

ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
LINK_BODY_NAMES = tuple(f"link{index}" for index in range(8)) + ("hand",)
HUMAN_MARKER_PREFIX = "human_"

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
class RobotLinkTrajectorySlice:
    timestamp: float
    link_positions_m: dict[str, tuple[float, float, float]]


@dataclass(frozen=True, slots=True)
class RobotLinkTrajectory:
    slices: tuple[RobotLinkTrajectorySlice, ...]


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
        self._kinematic_data = mujoco.MjData(self.model)
        self._human_mocap_ids: dict[str, int] = {}
        self._human_geom_ids: dict[str, int] = {}
        self._prediction_mocap_ids: dict[int, int] = {}
        self._prediction_geom_ids: dict[int, int] = {}
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name and name.startswith(HUMAN_MARKER_PREFIX):
                joint_name = name.removeprefix(HUMAN_MARKER_PREFIX)
                mocap_id = int(self.model.body_mocapid[body_id])
                if mocap_id >= 0:
                    self._human_mocap_ids[joint_name] = mocap_id
                geom_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_marker"
                )
                if geom_id >= 0:
                    self._human_geom_ids[joint_name] = geom_id
            if name and name.startswith("prediction_right_wrist_"):
                horizon_ms = int(name.rsplit("_", maxsplit=1)[1])
                mocap_id = int(self.model.body_mocapid[body_id])
                if mocap_id >= 0:
                    self._prediction_mocap_ids[horizon_ms] = mocap_id
                geom_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_marker"
                )
                if geom_id >= 0:
                    self._prediction_geom_ids[horizon_ms] = geom_id
        self.reset()

    def _name_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, object_type, name)
        if identifier < 0:
            raise ValueError(f"Required MuJoCo object is missing: {name}")
        return identifier

    def reset(self) -> RobotState:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._trajectory_time = 0.0
        self.model.opt.timestep = min(self.control_timestep_seconds, self.model.opt.timestep)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    @property
    def trajectory_time(self) -> float:
        return self._trajectory_time

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

    @property
    def human_marker_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._human_mocap_ids))

    def set_human_pose(self, frame: PoseFrame) -> None:
        if frame.frame_id != "world":
            raise ValueError("human pose must be expressed in the MuJoCo world frame")
        for joint_name, mocap_id in self._human_mocap_ids.items():
            observation = frame.joints.get(joint_name)
            geom_id = self._human_geom_ids.get(joint_name)
            if observation is None:
                if geom_id is not None:
                    self.model.geom_rgba[geom_id, 3] = 0.0
                continue
            self.data.mocap_pos[mocap_id] = observation.position_m
            if geom_id is not None:
                self.model.geom_rgba[geom_id, 3] = 1.0
        mujoco.mj_forward(self.model, self.data)

    def set_human_prediction(
        self,
        prediction: HumanPrediction,
        *,
        uncertainty_sigma: float = 2.0,
    ) -> None:
        if prediction.frame_id != "world":
            raise ValueError("human prediction must use the MuJoCo world frame")
        updated: set[int] = set()
        for prediction_slice in prediction.slices:
            horizon_ms = round(prediction_slice.horizon_seconds * 1000)
            predicted_joint = prediction_slice.joints.get("right_wrist")
            if predicted_joint is None or horizon_ms not in self._prediction_mocap_ids:
                continue
            self.data.mocap_pos[self._prediction_mocap_ids[horizon_ms]] = (
                predicted_joint.mean_position_m
            )
            geom_id = self._prediction_geom_ids.get(horizon_ms)
            if geom_id is not None:
                largest_variance = float(
                    np.linalg.eigvalsh(np.asarray(predicted_joint.covariance)).max()
                )
                self.model.geom_size[geom_id, 0] = max(
                    0.025, uncertainty_sigma * math.sqrt(max(0.0, largest_variance))
                )
                self.model.geom_rgba[geom_id, 3] = 0.22
            updated.add(horizon_ms)
        for horizon_ms, geom_id in self._prediction_geom_ids.items():
            if horizon_ms not in updated:
                self.model.geom_rgba[geom_id, 3] = 0.0
        mujoco.mj_forward(self.model, self.data)

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

    def planned_link_trajectory(
        self,
        times: tuple[float, ...],
        trajectory_times: tuple[float, ...] | None = None,
    ) -> RobotLinkTrajectory:
        if not times or any(timestamp < 0 for timestamp in times):
            raise ValueError("robot trajectory times must be non-empty and non-negative")
        planned_times = trajectory_times or times
        if len(planned_times) != len(times) or any(timestamp < 0 for timestamp in planned_times):
            raise ValueError("robot wall and trajectory times must be aligned and non-negative")
        slices: list[RobotLinkTrajectorySlice] = []
        for timestamp, trajectory_time in zip(times, planned_times, strict=True):
            mujoco.mj_resetDataKeyframe(self.model, self._kinematic_data, 0)
            self._kinematic_data.qpos[self._qpos_addresses] = self.desired_configuration(
                trajectory_time
            )
            mujoco.mj_forward(self.model, self._kinematic_data)
            links = {
                name: tuple(float(value) for value in self._kinematic_data.xpos[identifier])
                for name, identifier in self._link_ids.items()
            }
            slices.append(RobotLinkTrajectorySlice(timestamp=timestamp, link_positions_m=links))
        return RobotLinkTrajectory(slices=tuple(slices))

    def step(self, velocity_scale: float = 1.0) -> RobotState:
        if not 0 <= velocity_scale <= 1:
            raise ValueError("velocity scale must be between 0 and 1")
        desired = self.desired_configuration(self._trajectory_time)
        self.data.ctrl[self._actuator_ids] = desired
        target_time = self.data.time + self.control_timestep_seconds
        while self.data.time < target_time - 1e-12:
            mujoco.mj_step(self.model, self.data)
        self._trajectory_time += velocity_scale * self.control_timestep_seconds
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

        desired = self.desired_configuration(self._trajectory_time)
        actual = self.data.qpos[self._qpos_addresses]
        hand = tuple(float(value) for value in self.data.xpos[self._hand_id])
        return SimulationSummary(
            duration_seconds=float(self.data.time),
            physics_steps=steps,
            completed_moves=int(self._trajectory_time / self.move_duration_seconds),
            end_effector_path_length_m=path_length,
            final_joint_error_rad=float(np.linalg.norm(desired - actual)),
            final_end_effector_position_m=hand,
        )
