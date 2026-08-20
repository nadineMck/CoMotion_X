from pathlib import Path

from dotenv import dotenv_values


def test_example_environment_has_required_settings() -> None:
    values = dotenv_values(Path(".env.example"))

    assert values["COMOTION_X_ENV"] == "development"
    assert values["COMOTION_X_CONFIG"] == "config/default.toml"
    assert values["COMOTION_X_RESULTS_DIR"] == "results"
    assert values["COMOTION_X_DATA_DIR"] == "data"

