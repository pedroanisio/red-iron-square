"""Tests for the SDK CLI."""

import json
from pathlib import Path

from src.sdk.cli import main


class TestSdkCli:
    """Smoke tests for the CLI entrypoints."""

    def test_decide_command_prints_json(self, capsys) -> None:
        exit_code = main(
            [
                "decide",
                "--personality",
                '{"O":0.8,"C":0.5,"E":0.3,"A":0.7,"N":0.4,"R":0.9,"I":0.6,"T":0.2}',
                "--scenario",
                '{"name":"pitch_meeting","values":{"O":0.9,"N":0.7}}',
                "--actions",
                '[{"name":"bold","modifiers":{"O":1.0,"R":0.8,"N":-0.3}},{"name":"safe","modifiers":{"C":0.9,"T":0.8}}]',
            ]
        )

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["chosen_action"] in {"bold", "safe"}

    def test_simulate_self_aware_command_prints_json(self, capsys) -> None:
        exit_code = main(
            [
                "simulate",
                "--personality",
                '{"O":0.8,"C":0.5,"E":0.3,"A":0.7,"N":0.4,"R":0.9,"I":0.6,"T":0.2}',
                "--actions",
                '[{"name":"bold","modifiers":{"O":1.0,"R":0.8,"N":-0.3}},{"name":"safe","modifiers":{"C":0.9,"T":0.8}}]',
                "--scenarios",
                '[{"name":"pitch_meeting","values":{"O":0.9,"N":0.7}}]',
                "--outcomes",
                "[0.6]",
                "--self-model",
                '{"O":0.7,"C":0.5,"E":0.4,"A":0.6,"N":0.4,"R":0.8,"I":0.6,"T":0.3}',
            ]
        )

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["ticks"]) == 1
        assert "psi_hat" in payload["ticks"][0]

    def test_simulate_command_accepts_file_inputs(self, capsys) -> None:
        fixtures = Path(__file__).resolve().parent.parent / "examples"
        exit_code = main(
            [
                "simulate",
                "--personality",
                f"@{fixtures / 'personality.json'}",
                "--actions",
                f"@{fixtures / 'actions.json'}",
                "--scenarios",
                f"@{fixtures / 'scenarios.json'}",
                "--outcomes",
                "[0.4, 0.2, -0.1]",
                "--self-model",
                f"@{fixtures / 'self_model.json'}",
            ]
        )

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["ticks"]) == 3
        assert payload["ticks"][0]["scenario"]["name"] == "pitch_meeting"
