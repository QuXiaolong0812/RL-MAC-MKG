from dataclasses import replace

import yaml
from test_workflows import make_environment

from rl_mac_mkg.config import load_config
from rl_mac_mkg.project import load_dataset, write_json, write_jsonl


def test_selected_partition_does_not_disclose_heldout_entity_catalog(tmp_path):
    environment = make_environment()
    write_jsonl(
        tmp_path / "entities.jsonl",
        [entity.to_dict() for entity in environment.state.entities.values()],
    )
    write_jsonl(
        tmp_path / "sources.jsonl",
        [
            replace(source, partition="ppo").to_dict()
            for source in environment.state.sources.values()
        ],
    )
    write_json(tmp_path / "splits.json", {"ppo": ["a"], "test": ["b"]})
    write_json(tmp_path / "schema.json", environment.schema.to_dict())
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "format_version": 1,
                "dataset": {
                    "entities": "entities.jsonl",
                    "sources": "sources.jsonl",
                    "partition": "ppo",
                    "split_manifest": "splits.json",
                },
                "schema": {"path": "schema.json"},
                "reward": {"lambda_h": 1, "lambda_p": 1, "lambda_g": 1},
            }
        ),
        encoding="utf-8",
    )
    entities, _, seed_ids, _ = load_dataset(load_config(path))
    assert seed_ids == ["a"]
    assert [entity.entity_id for entity in entities] == ["a"]
