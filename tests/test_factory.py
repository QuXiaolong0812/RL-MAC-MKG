from dataclasses import replace

import yaml
from test_workflows import make_environment

from rl_mac_mkg.agents import CalibratedVerificationAgent
from rl_mac_mkg.config import load_config
from rl_mac_mkg.project import canonicalizer_for, fixed_specialists, write_json


def test_calibration_and_execution_use_same_partition_independent_scoring_signature(
    tmp_path, monkeypatch
):
    import rl_mac_mkg.backends.ranking as ranking
    import rl_mac_mkg.project as project

    class Backend:
        version = "fixed-backend"

    class CLIP:
        version = "fixed-clip"

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(project, "model_backend", lambda *args: Backend())
    monkeypatch.setattr(ranking, "CLIPBackend", CLIP)
    environment = make_environment()
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "format_version": 1,
                "reward": {"lambda_h": 1, "lambda_p": 1, "lambda_g": 1},
                "specialists": {
                    "clip": {"model_path": "clip"},
                    "calibration_file": "calibration.json",
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    calibration_entities = [
        replace(entity, entity_id="cal:" + entity.entity_id)
        for entity in environment.state.entities.values()
    ]
    calibration_verifier = CalibratedVerificationAgent(
        Backend(),
        environment.schema,
        1,
        1,
        canonicalizer=canonicalizer_for(config, calibration_entities),
    )
    write_json(
        tmp_path / "calibration.json",
        {
            "partition": "calibration",
            "verifier_version": calibration_verifier.score_version,
            "factual_temperature": 1.3,
            "visual_temperature": 1.8,
        },
    )
    specialists = fixed_specialists(
        config, list(environment.state.entities.values()), environment.schema
    )
    assert specialists.verifier.score_version == calibration_verifier.score_version
    assert specialists.evidence.min_image_size == (128, 128)
