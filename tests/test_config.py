from pathlib import Path

import pytest

from comotion_x.core.config import load_config


def test_default_config_loads() -> None:
    config = load_config(Path("config/default.toml"))

    assert config.project.name == "CoMotion-X"
    assert config.project.seed == 42
    assert config.robot.model_path.name == "scene.xml"
    assert config.robot.move_duration_seconds == 1.5
    assert config.prediction.horizons_seconds == (0.1, 0.2, 0.3, 0.5)
    assert config.safety.critical_distance_m < config.safety.warning_distance_m


def test_invalid_safety_distances_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(
        """
[project]
name = "test"
seed = 1
[simulation]
timestep_seconds = 0.01
duration_seconds = 1.0
[robot]
model_path = "model.xml"
move_duration_seconds = 1.0
[prediction]
horizons_seconds = [0.1]
[safety]
warning_distance_m = 0.2
critical_distance_m = 0.3
slow_velocity_scale = 0.5
""".strip()
    )

    with pytest.raises(ValueError, match="critical distance"):
        load_config(path)
