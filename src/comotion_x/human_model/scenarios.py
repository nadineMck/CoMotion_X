"""Synthetic upper-body motion scenarios in the MuJoCo world frame."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from comotion_x.core.models import JointObservation, PoseFrame, Vector3

JOINT_NAMES = (
    "torso",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)


class ScenarioName(StrEnum):
    STATIONARY = "stationary"
    SLOW_APPROACH = "slow_approach"
    SUDDEN_REACH = "sudden_reach"
    CROSSING = "crossing"
    NEAR_MISS = "near_miss"
    WITHDRAWAL = "withdrawal"
    OCCLUSION = "occlusion"
    VARIABLE_SPEED = "variable_speed"


@dataclass(frozen=True, slots=True)
class HumanScenario:
    name: ScenarioName
    seed: int
    frames_per_second: float
    ground_truth_frames: tuple[PoseFrame, ...]
    observation_frames: tuple[PoseFrame, ...]

    @property
    def duration_seconds(self) -> float:
        return self.ground_truth_frames[-1].timestamp

    def export_json(self, path: Path | str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "name": self.name.value,
            "seed": self.seed,
            "frame_id": "world",
            "units": {"position": "metres", "time": "seconds"},
            "frames_per_second": self.frames_per_second,
            "ground_truth_frames": [_frame_dict(frame) for frame in self.ground_truth_frames],
            "observation_frames": [_frame_dict(frame) for frame in self.observation_frames],
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _frame_dict(frame: PoseFrame) -> dict[str, Any]:
    return {
        "timestamp": round(frame.timestamp, 9),
        "joints": {
            name: {
                "position_m": [round(value, 9) for value in observation.position_m],
                "confidence": round(observation.confidence, 6),
            }
            for name, observation in sorted(frame.joints.items())
        },
    }


def _base_skeleton() -> dict[str, np.ndarray]:
    return {
        "torso": np.array([0.90, 0.35, 1.05]),
        "left_shoulder": np.array([0.90, 0.53, 1.28]),
        "right_shoulder": np.array([0.90, 0.17, 1.28]),
        "left_elbow": np.array([0.88, 0.66, 1.08]),
        "right_elbow": np.array([0.82, 0.08, 1.08]),
        "left_wrist": np.array([0.88, 0.72, 0.88]),
        "right_wrist": np.array([0.72, 0.02, 0.90]),
        "left_hip": np.array([0.92, 0.48, 0.82]),
        "right_hip": np.array([0.92, 0.22, 0.82]),
    }


def _smoothstep(value: float) -> float:
    clipped = min(max(value, 0.0), 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _lerp(start: np.ndarray, end: np.ndarray, amount: float) -> np.ndarray:
    return start + amount * (end - start)


def _ground_truth_pose(name: ScenarioName, progress: float) -> dict[str, np.ndarray]:
    pose = _base_skeleton()
    eased = _smoothstep(progress)

    if name is ScenarioName.STATIONARY:
        return pose
    if name is ScenarioName.SLOW_APPROACH:
        translation = np.array([-0.38 * eased, 0.0, 0.0])
        return {joint: position + translation for joint, position in pose.items()}
    if name is ScenarioName.WITHDRAWAL:
        initial_shift = np.array([-0.32, 0.0, 0.0])
        translation = initial_shift + np.array([0.58 * eased, 0.0, 0.0])
        return {joint: position + translation for joint, position in pose.items()}

    if name in {ScenarioName.SUDDEN_REACH, ScenarioName.OCCLUSION}:
        reach = _smoothstep((progress - 0.42) / 0.18)
        pose["right_elbow"] = _lerp(pose["right_elbow"], np.array([0.55, 0.04, 0.92]), reach)
        pose["right_wrist"] = _lerp(pose["right_wrist"], np.array([0.28, 0.00, 0.72]), reach)
        return pose

    if name is ScenarioName.CROSSING:
        pose["right_elbow"] = np.array([0.55, -0.30 + 0.60 * eased, 0.88])
        pose["right_wrist"] = np.array([0.42, -0.55 + 1.10 * eased, 0.72])
        return pose

    if name is ScenarioName.NEAR_MISS:
        pose["right_elbow"] = np.array([0.82, -0.25 + 0.55 * eased, 1.20])
        pose["right_wrist"] = np.array([0.72, -0.50 + 1.00 * eased, 1.32])
        return pose

    if name is ScenarioName.VARIABLE_SPEED:
        variable_progress = progress**3
        pose["right_elbow"] = _lerp(
            pose["right_elbow"], np.array([0.52, 0.02, 0.90]), variable_progress
        )
        pose["right_wrist"] = _lerp(
            pose["right_wrist"], np.array([0.30, -0.02, 0.73]), variable_progress
        )
        return pose

    raise ValueError(f"Unsupported scenario: {name}")


def generate_scenario(
    name: ScenarioName | str,
    *,
    duration_seconds: float = 4.0,
    frames_per_second: float = 30.0,
    seed: int = 42,
    noise_standard_deviation_m: float = 0.0,
    dropout_probability: float = 0.0,
) -> HumanScenario:
    scenario_name = ScenarioName(name)
    if duration_seconds <= 0 or frames_per_second <= 0:
        raise ValueError("scenario duration and frame rate must be positive")
    if seed < 0 or noise_standard_deviation_m < 0:
        raise ValueError("seed and observation noise must be non-negative")
    if not 0 <= dropout_probability <= 1:
        raise ValueError("dropout probability must be between 0 and 1")

    sample_count = round(duration_seconds * frames_per_second) + 1
    timestamps = np.linspace(0.0, duration_seconds, sample_count)
    random_generator = np.random.default_rng(seed)
    truth_frames: list[PoseFrame] = []
    observation_frames: list[PoseFrame] = []

    for timestamp in timestamps:
        progress = float(timestamp / duration_seconds)
        pose = _ground_truth_pose(scenario_name, progress)
        truth_joints = {
            joint: JointObservation(position_m=_vector(position), confidence=1.0)
            for joint, position in pose.items()
        }
        observed_joints: dict[str, JointObservation] = {}
        for joint, position in pose.items():
            scripted_occlusion = (
                scenario_name is ScenarioName.OCCLUSION
                and joint == "right_wrist"
                and 0.48 <= progress <= 0.68
            )
            random_dropout = random_generator.random() < dropout_probability
            if scripted_occlusion or random_dropout:
                continue
            noise = random_generator.normal(0.0, noise_standard_deviation_m, size=3)
            confidence = max(0.05, 1.0 - noise_standard_deviation_m * 10.0)
            observed_joints[joint] = JointObservation(
                position_m=_vector(position + noise), confidence=confidence
            )
        truth_frames.append(PoseFrame(float(timestamp), "world", truth_joints))
        observation_frames.append(PoseFrame(float(timestamp), "world", observed_joints))

    return HumanScenario(
        name=scenario_name,
        seed=seed,
        frames_per_second=frames_per_second,
        ground_truth_frames=tuple(truth_frames),
        observation_frames=tuple(observation_frames),
    )


def _vector(position: np.ndarray) -> Vector3:
    if position.shape != (3,) or not all(math.isfinite(float(value)) for value in position):
        raise ValueError("scenario produced an invalid 3D position")
    return tuple(float(value) for value in position)  # type: ignore[return-value]

