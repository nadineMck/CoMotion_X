"""Real-time camera pose, prediction, risk, and simulated-robot control loop."""

from __future__ import annotations

import json
import multiprocessing
import queue
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2

from comotion_x.core.config import AppConfig
from comotion_x.core.models import SafetyMode
from comotion_x.estimation.state_estimator import HumanStateEstimator
from comotion_x.evaluation.controllers import controller_parameters, occupancy_parameters
from comotion_x.perception.camera import CapturedFrame
from comotion_x.perception.pose_estimator import PoseDetection
from comotion_x.prediction.motion_predictor import HumanMotionPredictor
from comotion_x.robot.simulation import PandaSimulation
from comotion_x.safety.controller import PredictiveSafetyController
from comotion_x.safety.occupancy import current_clearance
from comotion_x.safety.risk import CollisionRiskEngine

OVERLAY_CONNECTIONS = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
)


class FrameSource(Protocol):
    def read(self) -> CapturedFrame | None: ...


class PoseDetector(Protocol):
    def detect(self, image_bgr, timestamp: float) -> PoseDetection | None: ...


@dataclass(frozen=True, slots=True)
class LiveCameraSummary:
    frames_read: int
    poses_detected: int
    elapsed_seconds: float
    capture_fps: float
    mean_inference_latency_ms: float
    pose_processing_fps: float
    minimum_measured_clearance_m: float | None
    final_safety_mode: SafetyMode
    final_velocity_scale: float
    recorded_pose_path: str | None

    def as_dict(self) -> dict[str, int | float | str | None]:
        return {
            "frames_read": self.frames_read,
            "poses_detected": self.poses_detected,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "capture_fps": round(self.capture_fps, 3),
            "mean_inference_latency_ms": round(self.mean_inference_latency_ms, 3),
            "pose_processing_fps": round(self.pose_processing_fps, 3),
            "minimum_measured_clearance_m": (
                round(self.minimum_measured_clearance_m, 6)
                if self.minimum_measured_clearance_m is not None
                else None
            ),
            "final_safety_mode": self.final_safety_mode.value,
            "final_velocity_scale": self.final_velocity_scale,
            "recorded_pose_path": self.recorded_pose_path,
        }


def run_live_camera(
    config: AppConfig,
    source: FrameSource,
    detector: PoseDetector,
    *,
    maximum_duration_seconds: float | None,
    display: bool = False,
    show_robot: bool = False,
    recorded_pose_path: Path | None = None,
) -> LiveCameraSummary:
    simulation = PandaSimulation(
        model_path=config.robot.model_path,
        control_timestep_seconds=config.simulation.timestep_seconds,
        move_duration_seconds=config.robot.move_duration_seconds,
    )
    estimator = HumanStateEstimator(
        observation_std_m=config.estimation.observation_noise_standard_deviation_m,
        acceleration_std_mps2=config.estimation.process_acceleration_standard_deviation_mps2,
        initial_velocity_std_mps=config.estimation.initial_velocity_standard_deviation_mps,
    )
    predictor = HumanMotionPredictor(
        acceleration_std_mps2=config.estimation.process_acceleration_standard_deviation_mps2
    )
    occupancy = occupancy_parameters(config)
    risk_engine = CollisionRiskEngine(occupancy)
    controller = PredictiveSafetyController(controller_parameters(config))
    velocity_scale = 1.0
    final_mode = SafetyMode.NORMAL
    last_pose = None
    frames_read = 0
    detections = 0
    latencies: list[float] = []
    minimum_clearance = float("inf")
    recorded_frames: list[dict] = []
    started = time.perf_counter()
    viewer = None
    display_process = None
    display_queue = None
    display_stop = None
    inline_display = display and not show_robot
    if display and show_robot:
        process_context = multiprocessing.get_context("spawn")
        display_queue = process_context.Queue(maxsize=2)
        display_stop = process_context.Event()
        display_process = process_context.Process(
            target=_camera_display_worker,
            args=(display_queue, display_stop),
            daemon=True,
        )
        display_process.start()
    if show_robot:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(simulation.model, simulation.data)
        viewer.cam.lookat[:] = (0.45, 0.15, 0.75)
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break
            if display_stop is not None and display_stop.is_set():
                break
            captured = source.read()
            if captured is None:
                break
            if (
                maximum_duration_seconds is not None
                and captured.timestamp > maximum_duration_seconds
            ):
                break
            frames_read += 1
            with viewer.lock() if viewer is not None else nullcontext():
                while simulation.data.time < captured.timestamp - 1e-9:
                    if last_pose is not None:
                        simulation.set_human_pose(last_pose)
                    simulation.step(velocity_scale)

            detection = detector.detect(captured.image_bgr, captured.timestamp)
            if detection is not None:
                detections += 1
                latencies.append(detection.inference_latency_seconds)
                last_pose = detection.pose_frame
                with viewer.lock() if viewer is not None else nullcontext():
                    simulation.set_human_pose(last_pose)
                    robot_state = simulation.state()
                measured_clearance = current_clearance(last_pose, robot_state, occupancy)
                minimum_clearance = min(minimum_clearance, measured_clearance)
                human_state = estimator.update(last_pose)
                prediction = predictor.predict(human_state, config.prediction.horizons_seconds)
                wall_times = tuple(item.timestamp for item in prediction.slices)
                trajectory_times = tuple(
                    simulation.trajectory_time + item.horizon_seconds
                    for item in prediction.slices
                )
                robot_trajectory = simulation.planned_link_trajectory(
                    wall_times, trajectory_times
                )
                predicted_risk = risk_engine.assess(prediction, robot_trajectory)
                with viewer.lock() if viewer is not None else nullcontext():
                    simulation.set_human_prediction(
                        prediction,
                        uncertainty_sigma=config.occupancy.uncertainty_sigma,
                    )
                decision = controller.update(
                    captured.timestamp,
                    current_clearance_m=measured_clearance,
                    predicted_risk=(predicted_risk if captured.timestamp >= 0.5 else None),
                )
                velocity_scale = decision.command.velocity_scale
                final_mode = decision.mode
                recorded_frames.append(_recorded_frame(detection))
                if display:
                    _draw_overlay(
                        captured.image_bgr,
                        detection,
                        mode=final_mode,
                        velocity_scale=velocity_scale,
                        clearance_m=predicted_risk.minimum_clearance_m,
                    )
            elif display:
                cv2.putText(
                    captured.image_bgr,
                    "POSE: NOT DETECTED",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            if inline_display:
                cv2.imshow("CoMotion-X Camera", captured.image_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif display_queue is not None:
                _queue_latest_frame(display_queue, captured.image_bgr)
            if viewer is not None:
                viewer.sync()
    finally:
        if inline_display:
            cv2.destroyAllWindows()
        if display_queue is not None:
            try:
                display_queue.put_nowait(None)
            except queue.Full:
                try:
                    display_queue.get_nowait()
                except queue.Empty:
                    pass
                display_queue.put_nowait(None)
        if display_process is not None:
            display_process.join(timeout=3.0)
            if display_process.is_alive():
                display_process.terminate()
                display_process.join(timeout=1.0)
        if viewer is not None:
            viewer.close()

    elapsed = time.perf_counter() - started
    if recorded_pose_path is not None:
        _write_pose_recording(recorded_pose_path, recorded_frames)
    total_inference = sum(latencies)
    return LiveCameraSummary(
        frames_read=frames_read,
        poses_detected=detections,
        elapsed_seconds=elapsed,
        capture_fps=frames_read / elapsed if elapsed > 0 else 0.0,
        mean_inference_latency_ms=(
            total_inference / len(latencies) * 1000 if latencies else 0.0
        ),
        pose_processing_fps=detections / total_inference if total_inference > 0 else 0.0,
        minimum_measured_clearance_m=(
            minimum_clearance if minimum_clearance != float("inf") else None
        ),
        final_safety_mode=final_mode,
        final_velocity_scale=velocity_scale,
        recorded_pose_path=str(recorded_pose_path) if recorded_pose_path is not None else None,
    )


def _recorded_frame(detection: PoseDetection) -> dict:
    return {
        "timestamp": detection.pose_frame.timestamp,
        "frame_id": detection.pose_frame.frame_id,
        "inference_latency_seconds": detection.inference_latency_seconds,
        "joints": {
            name: {
                "position_m": list(observation.position_m),
                "confidence": observation.confidence,
            }
            for name, observation in detection.pose_frame.joints.items()
        },
    }


def _write_pose_recording(path: Path, frames: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "mediapipe_pose_landmarker",
                "frame_id": "world",
                "frames": frames,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _draw_overlay(
    image,
    detection: PoseDetection,
    *,
    mode: SafetyMode,
    velocity_scale: float,
    clearance_m: float,
) -> None:
    height, width = image.shape[:2]
    pixels = {
        name: (int(point[0] * width), int(point[1] * height))
        for name, point in detection.image_points.items()
    }
    for first, second in OVERLAY_CONNECTIONS:
        if first in pixels and second in pixels:
            cv2.line(image, pixels[first], pixels[second], (40, 220, 255), 2)
    for point in pixels.values():
        cv2.circle(image, point, 5, (30, 80, 255), -1)
    status = (
        f"MODE: {mode.value.upper()}  SCALE: {velocity_scale:.2f}  "
        f"PRED CLEARANCE: {clearance_m:+.3f} m"
    )
    cv2.putText(
        image,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )


def _queue_latest_frame(display_queue, image) -> None:
    frame = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
    try:
        display_queue.put_nowait(frame)
    except queue.Full:
        try:
            display_queue.get_nowait()
        except queue.Empty:
            return
        try:
            display_queue.put_nowait(frame)
        except queue.Full:
            pass


def _camera_display_worker(display_queue, stop_event) -> None:
    while True:
        frame = display_queue.get()
        if frame is None:
            break
        cv2.imshow("CoMotion-X Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break
    cv2.destroyAllWindows()
