import pytest

from rl_mac_mkg.config import load_config


def test_configuration_requires_explicit_unreported_reward_weights(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("format_version: 1\nreward:\n  lambda_c: 0.08\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lambda_h"):
        load_config(path)


def test_configuration_paths_resolve_against_file_not_working_directory(tmp_path):
    path = tmp_path / "config" / "run.yaml"
    path.parent.mkdir()
    path.write_text(
        """format_version: 1
dataset:
  entities: ../entities.jsonl
  sources: ../sources.jsonl
reward:
  lambda_h: 1.0
  lambda_p: 0.5
  lambda_g: 0.2
  costs_file: ../costs.json
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.path("dataset.entities") == tmp_path / "entities.jsonl"
    assert config.path("reward.costs_file") == tmp_path / "costs.json"
    assert config.environment["budget"] == 30
    assert config.controller["lora_rank"] == 16


def test_configuration_rejects_unknown_keys_and_network_model_paths(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("format_version: 1\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
