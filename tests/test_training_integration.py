import json
from dataclasses import replace

from test_workflows import make_environment

from rl_mac_mkg.cli import main
from rl_mac_mkg.controller.checkpoint import load_controller
from rl_mac_mkg.environment import Environment
from rl_mac_mkg.project import write_json, write_jsonl
from rl_mac_mkg.workflows import FixedRoutePolicy, collect_episode


def test_bc_ppo_resume_and_inference_with_local_qwen(tmp_path, monkeypatch, capsys):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

    import rl_mac_mkg.cli as cli

    original = make_environment()
    sources = [replace(source, partition="ppo") for source in original.state.sources.values()]

    class Specialists:
        versions = original.specialist_versions

        def __call__(self, action, state):
            return original.dispatcher(action, state)

    monkeypatch.setattr(cli, "fixed_specialists", lambda *args: Specialists())
    base = tmp_path / "base"
    Qwen3Model(
        Qwen3Config(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    ).save_pretrained(base)
    tokenizer_engine = Tokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "a": 3, "b": 4}, unk_token="[UNK]")
    )
    tokenizer_engine.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_engine, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tokenizer.save_pretrained(base)
    environment = Environment(
        original.state.entities.values(),
        sources,
        original.schema,
        original.reward_config,
        Specialists(),
        budget=12,
        partition="ppo",
        seed_entity_ids=["a"],
        specialist_versions=Specialists.versions,
        reference_graph=[("a", "isA", "b")],
    )
    _, trace = collect_episode(environment, FixedRoutePolicy(), tokenizer)
    write_jsonl(
        tmp_path / "entities.jsonl",
        [entity.to_dict() for entity in original.state.entities.values()],
    )
    write_jsonl(tmp_path / "sources.jsonl", [source.to_dict() for source in sources])
    write_json(tmp_path / "schema.json", original.schema.to_dict())
    write_json(tmp_path / "splits.json", {"ppo": ["a"]})
    write_json(
        tmp_path / "costs.json",
        {"partition": "validation", "costs": original.reward_config.to_dict()["costs"]},
    )
    write_jsonl(
        tmp_path / "references.jsonl",
        [{"partition": "ppo", "entity_id": "a", "head": "a", "predicate": "isA", "tail": "b"}],
    )
    write_jsonl(
        tmp_path / "demonstrations.jsonl",
        [
            {
                "partition": "ppo",
                "entity_id": "a",
                "source_ids": ["s"],
                "context": row["context"],
                "operation": row["transition"]["action"]["operation"],
                "target_id": row["transition"]["action"]["target_id"],
            }
            for row in trace
        ],
    )
    import yaml

    config = {
        "format_version": 1,
        "dataset": {
            "entities": "entities.jsonl",
            "sources": "sources.jsonl",
            "partition": "ppo",
            "shared_entity_ids": ["b"],
            "split_manifest": "splits.json",
            "references": "references.jsonl",
            "demonstrations": "demonstrations.jsonl",
        },
        "schema": {"path": "schema.json"},
        "reward": {"lambda_h": 1, "lambda_p": 1, "lambda_g": 1, "costs_file": "costs.json"},
        "controller": {"model_path": "base", "head_size": 8},
        "ppo": {"batch_size": 4, "minibatch_size": 2, "epochs": 1},
        "environment": {"budget": 6, "max_tokens": 512},
        "run": {"episodes": 5, "seed": 8},
    }
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    bc = tmp_path / "bc"
    assert (
        main(
            ["train-controller", "--config", str(config_path), "--stage", "bc", "--output", str(bc)]
        )
        == 0
    ), capsys.readouterr().err
    ppo = tmp_path / "ppo"
    assert (
        main(
            [
                "train-controller",
                "--config",
                str(config_path),
                "--stage",
                "ppo",
                "--from-bc",
                str(bc),
                "--output",
                str(ppo),
            ]
        )
        == 0
    ), capsys.readouterr().err
    checkpoints = sorted(ppo.glob("update-*"))
    assert checkpoints
    model, metadata = load_controller(checkpoints[-1])
    assert model.policy_revision > 0
    assert metadata["metadata"]["stage"] == "ppo"
    inference_output = tmp_path / "inference"
    assert (
        main(
            [
                "infer",
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoints[-1]),
                "--seed-entity",
                "a",
                "--output",
                str(inference_output),
                "--deterministic",
            ]
        )
        == 0
    ), capsys.readouterr().err
    run = json.loads((inference_output / "run.json").read_text())
    assert run["steps"] >= 1
    assert len(checkpoints) > 1
    if len(checkpoints) > 1:
        resumed = tmp_path / "resumed"
        assert (
            main(
                [
                    "train-controller",
                    "--config",
                    str(config_path),
                    "--stage",
                    "ppo",
                    "--resume",
                    str(checkpoints[0]),
                    "--output",
                    str(resumed),
                ]
            )
            == 0
        ), capsys.readouterr().err
        assert list(resumed.glob("update-*"))
        config["reward"]["lambda_h"] = 99
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        assert (
            main(
                [
                    "train-controller",
                    "--config",
                    str(config_path),
                    "--stage",
                    "ppo",
                    "--resume",
                    str(checkpoints[0]),
                    "--output",
                    str(tmp_path / "changed"),
                ]
            )
            == 2
        )
        assert "reward settings changed" in capsys.readouterr().err
