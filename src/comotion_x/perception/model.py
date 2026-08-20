"""Download and verify the pinned official MediaPipe pose model."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
POSE_MODEL_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"


def model_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pose_model(path: Path | str) -> bool:
    model_path = Path(path)
    return model_path.is_file() and model_sha256(model_path) == POSE_MODEL_SHA256


def download_pose_model(path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    urllib.request.urlretrieve(POSE_MODEL_URL, temporary)  # noqa: S310
    if model_sha256(temporary) != POSE_MODEL_SHA256:
        raise RuntimeError("downloaded pose model failed SHA-256 verification")
    temporary.replace(destination)
    return destination

