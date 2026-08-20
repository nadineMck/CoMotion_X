"""Rigid transformation from MediaPipe body coordinates to MuJoCo world coordinates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from comotion_x.core.models import Vector3


@dataclass(frozen=True, slots=True)
class CameraWorldTransform:
    rotation: tuple[Vector3, Vector3, Vector3]
    translation_m: Vector3

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation_m, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("camera transform must contain a 3x3 rotation and 3D translation")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("camera transform values must be finite")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError("camera rotation must be orthonormal")
        if np.linalg.det(rotation) < 0.999:
            raise ValueError("camera rotation must be right-handed")

    @classmethod
    def from_json(cls, path: Path | str) -> CameraWorldTransform:
        payload = json.loads(Path(path).read_text())
        return cls(
            rotation=tuple(tuple(float(value) for value in row) for row in payload["rotation"]),  # type: ignore[arg-type]
            translation_m=tuple(float(value) for value in payload["translation_m"]),  # type: ignore[arg-type]
        )

    def transform(self, position: Vector3) -> Vector3:
        world = np.asarray(self.rotation) @ np.asarray(position) + np.asarray(self.translation_m)
        return tuple(float(value) for value in world)  # type: ignore[return-value]

