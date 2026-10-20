import json

import pytest

from rl_mac_mkg.cli import main


def test_cli_help_describes_real_training_workflows(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "train-controller" in help_text
    assert "train-specialist" in help_text
    assert "infer" in help_text


def test_evaluate_command_reads_predictions_without_model_dependencies(tmp_path, capsys):
    prediction = tmp_path / "predictions.jsonl"
    reference = tmp_path / "references.jsonl"
    row = {"head": "a", "predicate": "isA", "tail": "b", "kind": "relation"}
    prediction.write_text(json.dumps(row) + "\n", encoding="utf-8")
    reference.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert main(["evaluate", "--predictions", str(prediction), "--references", str(reference)]) == 0
    assert json.loads(capsys.readouterr().out)["relations"]["f1"] == 1


def test_cli_reports_missing_configuration_concisely(capsys):
    result = main(["validate", "--config", "does-not-exist.yaml"])
    assert result == 2
    assert "error" in capsys.readouterr().err.lower()
