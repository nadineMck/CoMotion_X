import csv
import json
from dataclasses import replace

from comotion_x.core.config import load_config
from comotion_x.evaluation.experiments import run_experiment_suite
from comotion_x.human_model.scenarios import ScenarioName


def test_small_experiment_writes_reproducible_artifacts(tmp_path) -> None:
    config = load_config("config/default.toml")
    short_config = replace(
        config,
        human=replace(config.human, duration_seconds=0.6, frames_per_second=10.0),
    )

    artifacts = run_experiment_suite(
        short_config,
        seeds=(3,),
        output_directory=tmp_path,
        scenario_names=(ScenarioName.STATIONARY,),
        controller_names=("unaware", "reactive"),
        include_prediction_study=False,
    )

    assert artifacts.trial_count == 2
    assert artifacts.timestep_count == 14
    assert all(path.is_file() for path in artifacts.files)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["seeds"] == [3]
    assert manifest["trial_count"] == 2
    with (tmp_path / "trial_metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["controller"] for row in rows} == {"unaware", "reactive"}
