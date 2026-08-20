"""MediaPipe Pose Landmarker adapter producing CoMotion-X world-frame poses."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from comotion_x.core.models import JointObservation, PoseFrame, Vector3
from comotion_x.perception.calibration import CameraWorldTransform
from comotion_x.perception.model import verify_pose_model

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "comotion-x-mpl"))

LANDMARK_INDICES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}


@dataclass(frozen=True, slots=True)
class PoseDetection:
    pose_frame: PoseFrame
    image_points: dict[str, tuple[float, float]]
    inference_latency_seconds: float


class MediaPipePoseEstimator:
    def __init__(
        self,
        model_path: Path | str,
        transform: CameraWorldTransform,
        *,
        minimum_confidence: float = 0.5,
    ) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum pose confidence must be between 0 and 1")
        if not verify_pose_model(model_path):
            raise FileNotFoundError(
                f"pose model missing or invalid: {model_path}; run prepare-camera-model"
            )
        import mediapipe as mp

        self._mp = mp
        self.transform = transform
        self.minimum_confidence = minimum_confidence
        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(Path(model_path).resolve()),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=minimum_confidence,
            min_pose_presence_confidence=minimum_confidence,
            min_tracking_confidence=minimum_confidence,
        )
        self._detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(
        self, image_bgr: NDArray[np.uint8], timestamp: float
    ) -> PoseDetection | None:
        if timestamp < 0:
            raise ValueError("camera timestamp must be non-negative")
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(round(timestamp * 1000), self._last_timestamp_ms + 1)
        started = time.perf_counter()
        result = self._detector.detect_for_video(image, timestamp_ms)
        latency = time.perf_counter() - started
        self._last_timestamp_ms = timestamp_ms
        return pose_detection_from_result(
            result,
            timestamp=timestamp,
            transform=self.transform,
            minimum_confidence=self.minimum_confidence,
            inference_latency_seconds=latency,
        )

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> MediaPipePoseEstimator:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def pose_detection_from_result(
    result: Any,
    *,
    timestamp: float,
    transform: CameraWorldTransform,
    minimum_confidence: float,
    inference_latency_seconds: float,
) -> PoseDetection | None:
    if not result.pose_world_landmarks or not result.pose_landmarks:
        return None
    world_landmarks = result.pose_world_landmarks[0]
    image_landmarks = result.pose_landmarks[0]
    joints: dict[str, JointObservation] = {}
    image_points: dict[str, tuple[float, float]] = {}
    for name, index in LANDMARK_INDICES.items():
        world_landmark = world_landmarks[index]
        image_landmark = image_landmarks[index]
        confidence = _confidence(world_landmark)
        if confidence < minimum_confidence:
            continue
        camera_position: Vector3 = (
            float(world_landmark.x),
            float(world_landmark.y),
            float(world_landmark.z),
        )
        joints[name] = JointObservation(
            position_m=transform.transform(camera_position), confidence=confidence
        )
        image_points[name] = (float(image_landmark.x), float(image_landmark.y))
    _add_midpoint("torso", "left_shoulder", "right_shoulder", joints, image_points)
    if not any(name in joints for name in ("torso", "left_wrist", "right_wrist")):
        return None
    return PoseDetection(
        pose_frame=PoseFrame(timestamp=timestamp, frame_id="world", joints=joints),
        image_points=image_points,
        inference_latency_seconds=inference_latency_seconds,
    )


def _confidence(landmark: Any) -> float:
    values = [
        float(value)
        for value in (getattr(landmark, "visibility", None), getattr(landmark, "presence", None))
        if value is not None
    ]
    return min(values) if values else 1.0


def _add_midpoint(
    name: str,
    first_name: str,
    second_name: str,
    joints: dict[str, JointObservation],
    image_points: dict[str, tuple[float, float]],
) -> None:
    if first_name not in joints or second_name not in joints:
        return
    first = joints[first_name]
    second = joints[second_name]
    midpoint = tuple(
        (a + b) * 0.5
        for a, b in zip(first.position_m, second.position_m, strict=True)
    )
    joints[name] = JointObservation(
        position_m=midpoint,  # type: ignore[arg-type]
        confidence=min(first.confidence, second.confidence),
    )
    first_image = image_points[first_name]
    second_image = image_points[second_name]
    image_points[name] = (
        (first_image[0] + second_image[0]) * 0.5,
        (first_image[1] + second_image[1]) * 0.5,
    )
