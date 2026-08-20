import json

from comotion_x.cli import main


def test_smoke_command_emits_valid_json(capsys) -> None:
    exit_code = main(["smoke", "--config", "config/default.toml"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["event"] == "smoke_check_passed"
    assert payload["seed"] == 42
    assert payload["safety_mode"] == "normal"

