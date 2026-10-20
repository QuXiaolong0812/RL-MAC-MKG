"""Configuration-driven loading of disjoint datasets, fixed experts and environments."""

import json
from pathlib import Path

from .config import ProjectConfig
from .data import audit_leakage, load_manifest
from .environment import Environment, RewardConfig
from .graph import Schema
from .records import Entity, stable_hash


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Each row must be an object")
            rows.append(row)
        except ValueError as error:
            raise ValueError(f"Invalid JSONL line {number}: {error}") from error
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_dataset(config: ProjectConfig, *, partition=None):
    partition = partition or config.values["dataset"]["partition"]
    entities = [Entity.from_dict(row) for row in read_jsonl(config.path("dataset.entities"))]
    if len({entity.entity_id for entity in entities}) != len(entities):
        raise ValueError("Entity IDs must be unique")
    sources = load_manifest(config.path("dataset.sources"))
    split_path = config.path("dataset.split_manifest", required=partition != "inference")
    splits = json.loads(split_path.read_text(encoding="utf-8")) if split_path else None
    if splits is not None:
        findings = audit_leakage(sources, splits)
        if findings:
            raise ValueError("Dataset leakage audit failed: " + "; ".join(findings))
        if any(
            identifier not in {entity.entity_id for entity in entities}
            for ids in splits.values()
            for identifier in ids
        ):
            raise ValueError("Split manifest references unknown entities")
    if partition != "inference":
        sources = [source for source in sources if source.partition == partition]
        seed_ids = splits.get(partition, [])
        shared_ids = config.values["dataset"]["shared_entity_ids"]
        if (
            not isinstance(shared_ids, list)
            or not all(isinstance(identifier, str) for identifier in shared_ids)
            or set(shared_ids) & {identifier for ids in splits.values() for identifier in ids}
        ):
            raise ValueError(
                "Shared ontology IDs must be explicit and absent from entity partitions"
            )
        allowed_ids = set(seed_ids) | set(shared_ids)
        entities = [entity for entity in entities if entity.entity_id in allowed_ids]
    else:
        seed_ids = sorted({identifier for source in sources for identifier in source.entity_ids})
    if not seed_ids:
        raise ValueError(f"No entities are assigned to {partition}")
    schema = Schema.from_dict(json.loads(config.path("schema.path").read_text(encoding="utf-8")))
    return entities, sources, seed_ids, schema


def model_backend(config, role):
    from .backends.http import HTTPBackend
    from .backends.local import LocalModelBackend

    settings = dict(config.values["specialists"].get(role, {}))
    if not settings:
        raise ValueError(f"Configuration requires specialists.{role}")
    backend = settings.pop("backend", "local")
    if backend == "http":
        for unsupported in ("model_path", "adapter_path", "dtype", "device", "multimodal"):
            settings.pop(unsupported, None)
        return HTTPBackend(**settings)
    settings["model_path"] = config.path(f"specialists.{role}.model_path")
    settings["adapter_path"] = config.path(f"specialists.{role}.adapter_path", required=False)
    settings.setdefault("multimodal", role in {"visual", "verifier"})
    return LocalModelBackend(**settings)


def canonicalizer_for(config, entities):
    from .backends.ranking import FrozenCanonicalizer, SentenceEncoder

    settings = config.values["specialists"].get("canonicalizer", {})
    encoder = (
        SentenceEncoder(
            config.path("specialists.canonicalizer.model_path"),
            device=settings.get("device", "cpu"),
        )
        if settings
        else None
    )
    return FrozenCanonicalizer(
        entities,
        encoder=encoder,
        threshold=settings.get("threshold", 0.95),
        ambiguity_margin=settings.get("ambiguity_margin", 0.05),
    )


def fixed_specialists(config, entities, schema):
    from .agents import (
        CalibratedVerificationAgent,
        EvidenceAgent,
        FixedSpecialists,
        GraphIntegrationAgent,
        TextualCandidateAgent,
        VisualGroundingAgent,
    )
    from .backends.ranking import CLIPBackend

    textual, visual, verifier = (
        model_backend(config, role) for role in ("textual", "visual", "verifier")
    )
    calibration = json.loads(
        config.path("specialists.calibration_file").read_text(encoding="utf-8")
    )
    if calibration.get("partition") != "calibration":
        raise ValueError("Calibration artifact does not identify the calibration partition")
    canonicalizer = canonicalizer_for(config, entities)
    clip = CLIPBackend(
        config.path("specialists.clip.model_path"),
        device=config.values["specialists"]["clip"].get("device", "cpu"),
    )
    media = config.values["media"]
    evidence = EvidenceAgent(
        schema=schema,
        clip_backend=clip,
        allowed_licenses=tuple(media["allowed_licenses"]),
        min_image_size=(media["minimum_resolution"],) * 2,
        max_hamming=media["phash_distance"],
        max_image_pixels=media["max_image_pixels"],
    )
    verification = CalibratedVerificationAgent(
        verifier,
        schema,
        calibration["factual_temperature"],
        calibration["visual_temperature"],
        canonicalizer=canonicalizer,
    )
    if calibration.get("verifier_version") != verification.score_version:
        raise ValueError(
            "Calibration artifact differs from the frozen verifier model, schema or prompts"
        )
    return FixedSpecialists(
        evidence,
        TextualCandidateAgent(textual, schema, canonicalizer=canonicalizer),
        VisualGroundingAgent(visual, schema, canonicalizer=canonicalizer),
        verification,
        GraphIntegrationAgent(schema, canonicalizer=canonicalizer),
    )


def environment_for(
    config, entities, sources, schema, specialists, seed_ids, *, partition, reference_graph=None
):
    reward = dict(config.values["reward"])
    reward.pop("costs_file", None)
    reward["tau_abs"] = reward.pop("abstention_threshold")
    costs_payload = json.loads(config.path("reward.costs_file").read_text(encoding="utf-8"))
    if costs_payload.get("partition") != "validation" or "costs" not in costs_payload:
        raise ValueError("Action cost artifact must identify validation measurements")
    reward_config = RewardConfig(costs=costs_payload["costs"], **reward)
    return Environment(
        entities,
        sources,
        schema,
        reward_config,
        specialists,
        budget=config.environment["budget"],
        max_depth=config.environment["max_depth"],
        specialist_versions=specialists.versions,
        partition=partition,
        seed_entity_ids=seed_ids,
        reference_graph=reference_graph,
        canonicalizer=getattr(getattr(specialists, "integration", None), "canonicalizer", None),
    )


def dataset_digest(entities, sources, schema):
    return stable_hash(
        {
            "entities": [entity.to_dict() for entity in entities],
            "sources": [
                {
                    "source_id": source.source_id,
                    "content_hash": source.content_hash,
                    "entity_ids": source.entity_ids,
                    "partition": source.partition,
                }
                for source in sources
            ],
            "schema": schema.to_dict(),
        }
    )


def export_episode(output, environment, trace, metadata=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Episode output already exists: {output}")
    output.mkdir(parents=True)
    write_json(output / "graph.json", environment.state.graph.to_dict())
    write_jsonl(
        output / "assertions.jsonl",
        [
            {
                **assertion.to_dict(),
                "kind": "image" if assertion.predicate in {"hasImage", "has_image"} else "relation",
            }
            for assertion in environment.state.graph.assertions.values()
        ],
    )
    write_jsonl(
        output / "candidates.jsonl",
        [candidate.to_dict() for candidate in environment.state.candidates.values()],
    )
    write_jsonl(
        output / "evidence.jsonl", [record.to_dict() for record in environment.state.evidence]
    )
    write_jsonl(output / "trace.jsonl", trace)
    write_json(
        output / "conflicts.json",
        [conflict.to_dict() for conflict in environment.state.conflicts.values()],
    )
    write_json(
        output / "run.json",
        {
            "budget": environment.state.initial_budget,
            "spent": environment.state.initial_budget - environment.state.budget,
            "depth": environment.state.depth,
            "steps": environment.state.step_count,
            "specialist_calls": sum(
                record.action.operation not in {"accept", "reject", "expand", "stop"}
                for record in environment.state.evidence
            ),
            "state_hash": environment.state.hash(),
            "specialist_versions": environment.specialist_versions,
            "partition": environment.partition,
            **(metadata or {}),
        },
    )
