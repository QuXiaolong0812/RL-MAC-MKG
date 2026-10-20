"""Command-line workflows for training, construction and reproducible evaluation."""

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import load_config
from .project import (
    canonicalizer_for,
    dataset_digest,
    environment_for,
    export_episode,
    fixed_specialists,
    load_dataset,
    model_backend,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .records import Entity, stable_hash


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _triple_keys(rows):
    result = []
    for row in rows:
        predicate = row.get("predicate", row.get("relation"))
        if not all(
            isinstance(value, str) and value
            for value in (row.get("head"), predicate, row.get("tail"))
        ):
            raise ValueError("Triples require nonempty head, predicate and tail strings")
        kind = row.get("kind", "image" if predicate in {"hasImage", "has_image"} else "relation")
        result.append((row["head"], predicate, row["tail"], kind))
    return result


def _tokenizer(model_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)


def _seed(value):
    import torch

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def command_validate(args):
    config = load_config(args.config)
    entities, sources, seed_ids, schema = load_dataset(config)
    result = {
        "entities": len(entities),
        "sources": len(sources),
        "seeds": len(seed_ids),
        "predicates": len(schema.predicates),
        "dataset_digest": dataset_digest(entities, sources, schema),
    }
    if args.models:
        specialists = fixed_specialists(config, entities, schema)
        result["specialist_versions"] = specialists.versions
    _print(result)


def command_split(args):
    from .data import (
        assign_sources,
        audit_leakage,
        duplicate_groups,
        image_duplicate_groups,
        load_manifest,
        split_entities,
    )

    entities = [Entity.from_dict(row) for row in read_jsonl(args.entities)]
    sources = load_manifest(args.sources)
    splits = split_entities([entity.entity_id for entity in entities], seed=args.seed)
    text_groups = duplicate_groups(
        {source.source_id: source.text for source in sources if source.kind == "text"}
    )
    image_groups = image_duplicate_groups(
        {source.source_id: source.path for source in sources if source.kind == "image"}
    )
    grouped = [
        replace(source, duplicate_group=(text_groups | image_groups).get(source.source_id))
        for source in sources
    ]
    assigned = assign_sources(grouped, splits)
    findings = audit_leakage(assigned, splits)
    if findings:
        raise ValueError("Split audit failed: " + "; ".join(findings))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Split output already exists: {output}")
    write_json(output / "splits.json", splits)
    write_jsonl(output / "sources.jsonl", [source.to_dict() for source in assigned])
    _print(
        {
            "partitions": {key: len(value) for key, value in splits.items()},
            "sources": len(assigned),
            "audit": findings,
        }
    )


def command_calibrate(args):
    from .evaluation.calibration import TemperatureScaler, calibration_metrics

    rows = read_jsonl(args.scores)
    if not rows or any(row.get("partition") != "calibration" for row in rows):
        raise ValueError("Calibration requires labeled calibration-partition scores")
    versions = {row.get("verifier_version") for row in rows}
    if len(versions) != 1 or not next(iter(versions)):
        raise ValueError("Calibration scores must identify one fixed verifier version")
    artifact = {
        "format_version": 1,
        "partition": "calibration",
        "verifier_version": next(iter(versions)),
        "score_digest": stable_hash(rows),
        "channels": {},
    }
    for channel in ("factual", "visual"):
        selected = [row for row in rows if row.get(channel + "_logit") is not None]
        if not selected:
            raise ValueError(f"Calibration has no {channel} labels")
        logits = [row[channel + "_logit"] for row in selected]
        labels = [row[channel + "_label"] for row in selected]
        scaler = TemperatureScaler.fit(logits, labels, partition="calibration")
        artifact[channel + "_temperature"] = scaler.temperature
        artifact["channels"][channel] = calibration_metrics(labels, scaler.predict(logits))
    write_json(args.output, artifact)
    _print(artifact)


def command_score_verifier(args):
    from .agents import CalibratedVerificationAgent
    from .environment import ConstructionState
    from .graph import GraphStore
    from .records import Candidate

    config = load_config(args.config)
    entities, sources, seed_ids, schema = load_dataset(config, partition="calibration")
    backend = model_backend(config, "verifier")
    verifier = CalibratedVerificationAgent(
        backend, schema, 1.0, 1.0, canonicalizer=canonicalizer_for(config, entities)
    )
    state = ConstructionState(
        {entity.entity_id: entity for entity in entities},
        {source.source_id: source for source in sources},
        GraphStore(),
        {},
        0,
        0,
        config.environment["max_depth"],
        tuple(seed_ids),
    )
    output = []
    for row in read_jsonl(args.candidates):
        if row.get("partition") != "calibration" or row.get("entity_id") not in seed_ids:
            raise ValueError("Verifier scoring requires calibration candidates")
        candidate = Candidate.from_dict(row["candidate"])
        scores = verifier.raw_scores(candidate, state)
        output.append(
            {
                "partition": "calibration",
                "entity_id": row["entity_id"],
                "candidate_id": candidate.candidate_id,
                "verifier_version": verifier.score_version,
                **scores,
                "factual_label": row["factual_label"],
                "visual_label": row.get("visual_label"),
            }
        )
    write_jsonl(args.output, output)
    _print({"candidates": len(output), "verifier_version": verifier.score_version})


def command_evaluate_retrieval(args):
    from .evaluation.metrics import retrieval_metrics

    rankings = json.loads(Path(args.rankings).read_text(encoding="utf-8"))
    relevance = json.loads(Path(args.relevance).read_text(encoding="utf-8"))
    _print(
        retrieval_metrics(rankings, {key: set(value) for key, value in relevance.items()}, k=args.k)
    )


def command_perturb(args):
    from .data import load_manifest
    from .evaluation.perturbations import perturb_sources

    sources = load_manifest(args.sources)
    edits = (
        json.loads(Path(args.replacements).read_text(encoding="utf-8"))
        if args.replacements
        else None
    )
    conflicts = load_manifest(args.conflicts) if args.conflicts else None
    perturbed, audit = perturb_sources(
        sources,
        args.condition,
        rate=args.rate,
        seed=args.seed,
        replacements=edits,
        conflicting_sources=conflicts,
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Perturbation output already exists: {output}")
    write_jsonl(output / "sources.jsonl", [source.to_dict() for source in perturbed])
    write_json(output / "perturbation.json", audit)
    _print(audit)


def command_fit_costs(args):
    from .workflows import fit_action_costs

    rows = read_jsonl(args.measurements)
    costs = fit_action_costs(
        rows,
        token_weight=args.token_weight,
        latency_weight=args.latency_weight,
        retrieval_weight=args.retrieval_weight,
    )
    artifact = {
        "format_version": 1,
        "partition": "validation",
        "costs": costs,
        "measurement_digest": stable_hash(rows),
        "weights": {
            "tokens": args.token_weight,
            "latency": args.latency_weight,
            "retrieval": args.retrieval_weight,
        },
    }
    write_json(args.output, artifact)
    _print(artifact)


def command_train_specialist(args):
    from .training.supervised import fit_specialist

    config = load_config(args.config)
    partition = "verifier" if args.role == "verifier" else "specialist"
    entities, sources, seed_ids, _ = load_dataset(config, partition=partition)
    settings = config.values["specialists"][args.role]
    if settings.get("backend", "local") != "local":
        raise ValueError("Specialist fitting requires a local model")
    result = fit_specialist(
        config.path(f"specialists.{args.role}.model_path"),
        read_jsonl(args.examples),
        {source.source_id: source for source in sources},
        args.output,
        partition=partition,
        entity_ids=set(seed_ids),
        multimodal=settings.get("multimodal", args.role != "textual"),
        device=settings.get("device", "cpu"),
        dtype=settings.get("dtype", "float32"),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        accumulation_steps=args.accumulation_steps,
        max_tokens=config.environment["max_tokens"],
        seed=config.values["run"]["seed"],
    )
    _print(result)


def command_infer(args):
    from .controller.checkpoint import load_controller
    from .workflows import FixedRoutePolicy, collect_episode

    config = load_config(args.config)
    _seed(config.values["run"]["seed"])
    entities, sources, seed_ids, schema = load_dataset(config)
    if args.seed_entity:
        if args.seed_entity not in seed_ids:
            raise ValueError("Requested seed is outside the selected dataset partition")
        seed_ids = [args.seed_entity]
    specialists = fixed_specialists(config, entities, schema)
    partition = config.values["dataset"]["partition"]
    environment = environment_for(
        config, entities, sources, schema, specialists, seed_ids, partition=partition
    )
    if args.policy == "fixed":
        policy = FixedRoutePolicy(config.values["reward"]["abstention_threshold"])
        tokenizer = _tokenizer(config.path("controller.model_path"))
    else:
        checkpoint = args.checkpoint or config.path("controller.checkpoint")
        policy, checkpoint_config = load_controller(
            checkpoint,
            base_path=config.path("controller.model_path"),
            device=config.controller["device"],
            trainable=False,
        )
        expected = checkpoint_config["metadata"].get("specialist_signature")
        if expected != stable_hash(specialists.versions):
            raise ValueError("Checkpoint specialists differ from the inference configuration")
        tokenizer_path = Path(checkpoint) / "tokenizer"
        tokenizer = _tokenizer(
            tokenizer_path if tokenizer_path.exists() else config.path("controller.model_path")
        )
    _, trace = collect_episode(
        environment,
        policy,
        tokenizer,
        max_tokens=config.environment["max_tokens"],
        deterministic=args.deterministic,
    )
    output = args.output or config.path("run.output")
    export_episode(
        output, environment, trace, {"policy": args.policy, "seed": config.values["run"]["seed"]}
    )
    _print(
        {
            "output": str(Path(output).resolve()),
            "assertions": len(environment.state.graph.assertions),
            "steps": environment.state.step_count,
            "remaining_budget": environment.state.budget,
        }
    )


def command_train_controller(args):
    import torch

    from .controller.checkpoint import load_controller, restore_trainer, save_controller
    from .controller.context import DecisionContext
    from .controller.policy import NeuralController
    from .environment import ConstructionState
    from .training.ppo import PPOConfig, PPOTrainer, behavioral_cloning_update
    from .workflows import collect_episode, observation_for

    config = load_config(args.config)
    if args.epochs < 1:
        raise ValueError("Controller epochs must be positive")
    _seed(config.values["run"]["seed"])
    entities, sources, seed_ids, schema = load_dataset(config, partition="ppo")
    specialists = fixed_specialists(config, entities, schema)
    signature = stable_hash(specialists.versions)
    digest = dataset_digest(entities, sources, schema)
    base = config.path("controller.model_path")
    tokenizer = _tokenizer(base)
    checkpoint_metadata = {}
    if args.stage == "bc":
        options = dict(config.controller)
        options.pop("model_path", None)
        options.pop("checkpoint", None)
        options["rank"] = options.pop("lora_rank")
        options["alpha"] = options.pop("lora_alpha")
        options["target_modules"] = options.pop("lora_target_modules")
        model = NeuralController.from_local(base, **options)
        rows = read_jsonl(args.demonstrations or config.path("dataset.demonstrations"))
        demonstrations = []
        for row in rows:
            if (
                row.get("partition") != "ppo"
                or row.get("entity_id") not in seed_ids
                or not row.get("source_ids")
                or not set(row["source_ids"]) <= {source.source_id for source in sources}
            ):
                raise ValueError("Behavioral cloning trace crosses PPO data partitions")
            context = DecisionContext.from_dict(row["context"])
            if context.specialist_signature != signature:
                raise ValueError(
                    "Behavioral cloning specialists differ from configured specialists"
                )
            operation = context.operation_names.index(row["operation"])
            target = context.object_ids.index(row.get("target_id"))
            demonstrations.append((context, operation, target))
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.values["ppo"]["learning_rate"],
            weight_decay=0.0,
        )
        losses = []
        for _ in range(args.epochs):
            random.shuffle(demonstrations)
            losses.append(
                behavioral_cloning_update(
                    model,
                    optimizer,
                    demonstrations,
                    minibatch_size=config.values["ppo"]["minibatch_size"],
                )
            )
        save_controller(
            model,
            args.output,
            base_path=base,
            tokenizer=tokenizer,
            metadata={
                "stage": "bc",
                "specialist_signature": signature,
                "dataset_digest": digest,
                "losses": losses,
            },
        )
        _print({"stage": "bc", "losses": losses, "checkpoint": str(Path(args.output).resolve())})
        return
    initial = args.resume or args.from_bc
    if not initial:
        raise ValueError("PPO requires --from-bc or --resume")
    model, checkpoint_config = load_controller(
        initial, base_path=base, device=config.controller["device"]
    )
    checkpoint_metadata = checkpoint_config["metadata"]
    required_stage = "ppo" if args.resume else "bc"
    if checkpoint_metadata.get("stage") != required_stage:
        raise ValueError(f"Expected a {required_stage} initialization checkpoint")
    if (
        checkpoint_metadata.get("specialist_signature") != signature
        or checkpoint_metadata.get("dataset_digest") != digest
    ):
        raise ValueError("Training checkpoint dataset or specialists changed")
    if model.temperature != config.controller["temperature"]:
        raise ValueError("Controller temperature differs from checkpoint")
    trainer = PPOTrainer(model, PPOConfig(**config.values["ppo"]))
    if args.resume:
        restore_trainer(trainer, initial)
    reference_rows = read_jsonl(config.path("dataset.references"))
    if any(
        row.get("partition") != "ppo" or row.get("entity_id") not in seed_ids
        for row in reference_rows
    ):
        raise ValueError("Terminal annotations must belong exclusively to PPO training entities")
    references = {}
    for row in reference_rows:
        references.setdefault(row["entity_id"], set()).add(tuple(_triple_keys([row])[0][:3]))
    if any(seed not in references for seed in seed_ids):
        raise ValueError("Every PPO seed needs an annotated reference graph")
    training_signature = stable_hash(
        {
            "dataset": digest,
            "references": reference_rows,
            "splits": json.loads(config.path("dataset.split_manifest").read_text(encoding="utf-8")),
            "seed_ids": seed_ids,
            "environment": config.environment,
            "reward": config.values["reward"],
            "costs": json.loads(config.path("reward.costs_file").read_text(encoding="utf-8")),
            "context_format_version": 1,
        }
    )
    if args.resume and checkpoint_metadata.get("training_signature") != training_signature:
        raise ValueError(
            "Resume training annotations, partitions, environment or reward settings changed"
        )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Training output already exists: {output}")
    output.mkdir(parents=True)
    episodes = config.values["run"]["episodes"]
    schedule = checkpoint_metadata.get("seed_schedule") or [
        seed_ids[index % len(seed_ids)] for index in range(episodes)
    ]
    index = checkpoint_metadata.get("next_episode", 0) if args.resume else 0
    environment = None
    if args.resume and checkpoint_metadata.get("pending_state"):
        seed = schedule[index]
        environment = environment_for(
            config,
            entities,
            sources,
            schema,
            specialists,
            [seed],
            partition="ppo",
            reference_graph=references[seed],
        )
        environment.state = ConstructionState.from_dict(checkpoint_metadata["pending_state"])
    batch, reports = [], []
    trace_rows = checkpoint_metadata.get("pending_trace", []) if args.resume else []
    while index < len(schedule):
        seed = schedule[index]
        if environment is None:
            environment = environment_for(
                config,
                entities,
                sources,
                schema,
                specialists,
                [seed],
                partition="ppo",
                reference_graph=references[seed],
            )
        steps, trace = collect_episode(
            environment,
            model,
            tokenizer,
            training=True,
            max_tokens=config.environment["max_tokens"],
            max_steps=trainer.config.batch_size - len(batch),
        )
        batch.extend(steps)
        trace_rows.extend({"episode": index, "seed_entity": seed, **row} for row in trace)
        with (output / "rollouts.jsonl").open("a", encoding="utf-8") as stream:
            for row in trace:
                stream.write(
                    json.dumps({"episode": index, "seed_entity": seed, **row}, ensure_ascii=False)
                    + "\n"
                )
        finished = environment.state.done
        if finished:
            export_episode(
                output / f"episode-{index:05d}",
                environment,
                [row for row in trace_rows if row["episode"] == index],
                {"seed_entity": seed},
            )
            index += 1
        if len(batch) == trainer.config.batch_size or index == len(schedule):
            bootstrap = None
            if batch and not batch[-1].done:
                with torch.no_grad():
                    _, value = model.distribution(
                        observation_for(
                            environment, tokenizer, max_tokens=config.environment["max_tokens"]
                        )
                    )
                    bootstrap = float(value.item())
            report = trainer.update(
                batch, bootstrap_value=bootstrap, expected_specialists=signature
            )
            reports.append(report)
            save_controller(
                model,
                output / f"update-{trainer.updates:06d}",
                base_path=base,
                tokenizer=tokenizer,
                trainer=trainer,
                metadata={
                    "stage": "ppo",
                    "specialist_signature": signature,
                    "dataset_digest": digest,
                    "next_episode": index,
                    "seed_schedule": schedule,
                    "training_signature": training_signature,
                    "pending_state": None if finished else environment.state.to_dict(),
                    "pending_trace": [] if finished else trace_rows,
                },
            )
            batch = []
            write_json(output / "training.json", reports)
        if finished:
            environment = None
            trace_rows = []
    _print({"updates": trainer.updates, "episodes": len(schedule), "output": str(output.resolve())})


def command_evaluate(args):
    from .evaluation.metrics import construction_metrics

    _print(
        construction_metrics(
            _triple_keys(read_jsonl(args.predictions)), _triple_keys(read_jsonl(args.references))
        )
    )


def command_statistics(args):
    from .evaluation.statistics import holm_correction, paired_comparison

    rows = read_jsonl(args.results)
    groups = {}
    for row in rows:
        if not row.get("entity_id") or not row.get("comparison"):
            raise ValueError("Statistical rows require aligned entity IDs and a comparison name")
        groups.setdefault(row["comparison"], []).append(row)
    reports = {}
    for name, values in groups.items():
        if len({value["entity_id"] for value in values}) != len(values):
            raise ValueError("Duplicate entity pairs in statistical comparison")
        reports[name] = paired_comparison(
            [row["first"] for row in values],
            [row["second"] for row in values],
            resamples=args.resamples,
            seed=args.seed,
        )
    adjusted = holm_correction([report["p_value"] for report in reports.values()])
    for report, value in zip(reports.values(), adjusted, strict=True):
        report["holm_p_value"] = value
    if args.output:
        write_json(args.output, reports)
    _print(reports)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="rl-mac-mkg", description="PPO-coordinated multimodal knowledge graph construction"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Audit dataset partitions and source integrity")
    validate.add_argument("--config", required=True)
    validate.add_argument("--models", action="store_true")
    validate.set_defaults(function=command_validate)
    split = commands.add_parser("split", help="Create entity/source-disjoint partitions")
    for name in ("entities", "sources", "output"):
        split.add_argument("--" + name, required=True)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(function=command_split)
    calibrate = commands.add_parser(
        "calibrate", help="Fit verifier temperatures on calibration labels"
    )
    calibrate.add_argument("--scores", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.set_defaults(function=command_calibrate)
    score = commands.add_parser(
        "score-verifier", help="Collect conditional verifier logits on calibration candidates"
    )
    for name in ("config", "candidates", "output"):
        score.add_argument("--" + name, required=True)
    score.set_defaults(function=command_score_verifier)
    costs = commands.add_parser("fit-costs", help="Fit action costs from validation measurements")
    for name in ("measurements", "output"):
        costs.add_argument("--" + name, required=True)
    for name in ("token-weight", "latency-weight", "retrieval-weight"):
        costs.add_argument("--" + name, type=float, required=True)
    costs.set_defaults(function=command_fit_costs)
    specialist = commands.add_parser(
        "train-specialist", help="Fit a specialist LoRA adapter before PPO"
    )
    for name in ("config", "examples", "output"):
        specialist.add_argument("--" + name, required=True)
    specialist.add_argument("--role", choices=["textual", "visual", "verifier"], required=True)
    specialist.add_argument("--epochs", type=int, default=1)
    specialist.add_argument("--learning-rate", type=float, default=1e-5)
    specialist.add_argument("--accumulation-steps", type=int, default=16)
    specialist.set_defaults(function=command_train_specialist)
    controller = commands.add_parser(
        "train-controller", help="Behaviorally clone or PPO-train the controller"
    )
    for name in ("config", "output"):
        controller.add_argument("--" + name, required=True)
    controller.add_argument("--stage", choices=["bc", "ppo"], required=True)
    controller.add_argument("--demonstrations")
    controller.add_argument("--from-bc")
    controller.add_argument("--resume")
    controller.add_argument("--epochs", type=int, default=1)
    controller.set_defaults(function=command_train_controller)
    infer = commands.add_parser("infer", help="Construct a graph using PPO or fixed routing")
    infer.add_argument("--config", required=True)
    infer.add_argument("--checkpoint")
    infer.add_argument("--output")
    infer.add_argument("--seed-entity")
    infer.add_argument("--policy", choices=["ppo", "fixed"], default="ppo")
    infer.add_argument("--deterministic", action="store_true")
    infer.set_defaults(function=command_infer)
    evaluate = commands.add_parser("evaluate", help="Score relation and image construction")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--references", required=True)
    evaluate.set_defaults(function=command_evaluate)
    retrieval = commands.add_parser(
        "evaluate-retrieval", help="Evaluate a fixed retriever's ranked results"
    )
    for name in ("rankings", "relevance"):
        retrieval.add_argument("--" + name, required=True)
    retrieval.add_argument("--k", type=int, default=10)
    retrieval.set_defaults(function=command_evaluate_retrieval)
    perturb = commands.add_parser("perturb", help="Apply audited evidence perturbations")
    for name in ("sources", "output", "condition"):
        perturb.add_argument("--" + name, required=True)
    perturb.add_argument("--rate", type=float, required=True)
    perturb.add_argument("--seed", type=int, default=42)
    perturb.add_argument("--replacements")
    perturb.add_argument("--conflicts")
    perturb.set_defaults(function=command_perturb)
    statistics = commands.add_parser(
        "statistics", help="Paired entity bootstrap/permutation and Holm correction"
    )
    statistics.add_argument("--results", required=True)
    statistics.add_argument("--output")
    statistics.add_argument("--resamples", type=int, default=10000)
    statistics.add_argument("--seed", type=int, default=42)
    statistics.set_defaults(function=command_statistics)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.function(args)
        return 0
    except (ValueError, OSError, KeyError, ImportError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
