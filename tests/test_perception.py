from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from comotion_x.core.config import load_config
from comotion_x.human_model.scenarios import generate_scenario
from comotion_x.perception.calibration import CameraWorldTransform
from comotion_x.perception.camera import CapturedFrame, OpenCVFrameSource
from comotion_x.perception.live import run_live_camera
from comotion_x.perception.pose_estimator import PoseDetection, pose_detection_from_result


@dataclass
class Landmark:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0


@dataclass
class Result:
    pose_world_landmarks: list[list[Landmark]]
    pose_landmarks: list[list[Landmark]]


def test_camera_world_transform_loads_and_transforms() -> None:
    transform = CameraWorldTransform.from_json("config/camera_to_world.json")

    point = transform.transform((0.1, -0.2, 0.3))

    assert point == pytest.approx((0.6, 0.45, 1.02))


def test_mediapipe_result_maps_selected_joints_and_confidence() -> None:
    world = [Landmark() for _ in range(33)]
    image = [Landmark(x=0.5, y=0.5) for _ in range(33)]
    world[11] = Landmark(x=-0.2, y=-0.4, z=0.1, visibility=0.9, presence=0.8)
    world[12] = Landmark(x=0.2, y=-0.4, z=0.1, visibility=0.9, presence=0.8)
    world[15] = Landmark(x=-0.4, y=0.0, z=0.0, visibility=0.2, presence=1.0)
    transform = CameraWorldTransform.from_json("config/camera_to_world.json")

    detection = pose_detection_from_result(
        Result([world], [image]),
        timestamp=1.25,
        transform=transform,
        minimum_confidence=0.5,
        inference_latency_seconds=0.012,
    )

    assert detection is not None
    assert detection.pose_frame.frame_id == "world"
    assert detection.pose_frame.timestamp == 1.25
    assert "torso" in detection.pose_frame.joints
    assert "left_wrist" not in detection.pose_frame.joints
    assert detection.pose_frame.joints["left_shoulder"].confidence == pytest.approx(0.8)


def test_recorded_video_source_produces_monotonic_frames(tmp_path) -> None:
    video_path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 24),
    )
    assert writer.isOpened()
    for value in (0, 80, 160):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()

    with OpenCVFrameSource(video_path) as source:
        frames = [source.read(), source.read(), source.read()]
        end = source.read()

    assert all(frame is not None for frame in frames)
    timestamps = [frame.timestamp for frame in frames if frame is not None]
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == 3
    assert end is None


class FakeSource:
    def __init__(self, frames: list[CapturedFrame]) -> None:
        self.frames = iter(frames)

    def read(self) -> CapturedFrame | None:
        return next(self.frames, None)


class FakeDetector:
    def __init__(self, detections: dict[float, PoseDetection]) -> None:
        self.detections = detections

    def detect(self, _image, timestamp: float) -> PoseDetection | None:
        return self.detections.get(timestamp)


def test_live_pipeline_records_poses_without_physical_camera(tmp_path) -> None:
    config = load_config("config/default.toml")
    scenario = generate_scenario("crossing", duration_seconds=0.6, frames_per_second=10.0)
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    frames = [
        CapturedFrame(image_bgr=image.copy(), timestamp=frame.timestamp, index=index)
        for index, frame in enumerate(scenario.observation_frames)
    ]
    detections = {
        frame.timestamp: PoseDetection(
            pose_frame=frame,
            image_points={},
            inference_latency_seconds=0.01,
        )
        for frame in scenario.observation_frames
    }
    output = tmp_path / "poses.json"

    summary = run_live_camera(
        config,
        FakeSource(frames),
        FakeDetector(detections),
        maximum_duration_seconds=0.6,
        recorded_pose_path=output,
    )

    assert summary.frames_read == 7
    assert summary.poses_detected == 7
    assert summary.mean_inference_latency_ms == pytest.approx(10.0)
    assert output.is_file()


def test_live_pipeline_without_duration_runs_until_source_closes() -> None:
    config = load_config("config/default.toml")
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    frames = [
        CapturedFrame(image_bgr=image.copy(), timestamp=timestamp, index=index)
        for index, timestamp in enumerate((0.0, 60.0, 120.0))
    ]

    summary = run_live_camera(
        config,
        FakeSource(frames),
        FakeDetector({}),
        maximum_duration_seconds=None,
    )

    assert summary.frames_read == 3
