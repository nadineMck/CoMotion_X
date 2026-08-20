"""OpenCV webcam and recorded-video frame source."""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    image_bgr: NDArray[np.uint8]
    timestamp: float
    index: int


class OpenCVFrameSource:
    def __init__(
        self,
        source: int | Path | str,
        *,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.is_camera = isinstance(source, int)
        resolved_source: int | str = source if isinstance(source, int) else str(source)
        self.capture = cv2.VideoCapture(resolved_source)
        if not self.capture.isOpened():
            kind = "camera" if self.is_camera else "video"
            hint = (
                " Grant camera access in System Settings > Privacy & Security > Camera."
                if self.is_camera and platform.system() == "Darwin"
                else ""
            )
            raise RuntimeError(f"unable to open {kind} source: {source}.{hint}")
        if self.is_camera:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._start_time = time.perf_counter()
        self._index = 0
        self._last_timestamp = -1.0
        self._fps = self.capture.get(cv2.CAP_PROP_FPS)

    def read(self) -> CapturedFrame | None:
        success, image = self.capture.read()
        if not success:
            return None
        if self.is_camera:
            timestamp = time.perf_counter() - self._start_time
        else:
            timestamp = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp <= self._last_timestamp:
                fps = self._fps if self._fps > 0 else 30.0
                timestamp = self._index / fps
        timestamp = max(timestamp, self._last_timestamp + 1e-6)
        frame = CapturedFrame(image_bgr=image, timestamp=timestamp, index=self._index)
        self._index += 1
        self._last_timestamp = timestamp
        return frame

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> OpenCVFrameSource:
        return self

    def __exit__(self, *_args) -> None:
        self.close()
