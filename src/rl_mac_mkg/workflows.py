"""Reproducible observation packing, rollout execution and resource-cost fitting."""

import math
from collections import Counter

from .controller.context import build_context
from .records import Action, Operation, stable_hash


def observation_for(environment, tokenizer, *, max_tokens=2048):
    state = environment.state
    actions = environment.available_actions()
    summaries = []
    unresolved = sorted(
        (candidate for candidate in state.candidates.values() if candidate.unresolved),
        key=lambda candidate: (candidate.created_step, candidate.candidate_id),
    )
    for candidate in unresolved:
        verification = state.verifications.get(candidate.candidate_id)
        conflicts = [
            conflict.to_dict()
            for conflict in state.conflicts.values()
            if not conflict.resolved and candidate.candidate_id in conflict.candidate_ids
        ]
        details = {
            "head": candidate.head,
            "predicate": candidate.predicate,
            "tail": candidate.tail,
            "version": candidate.version,
            "status": candidate.status.value,
            "verification": verification.to_dict() if verification else None,
            "age": state.step_count - candidate.created_step,
            "conflicts": conflicts,
            "source_ids": [anchor.source_id for anchor in candidate.anchors],
            "recent_calls": [
                (record.action.operation.value, record.status, record.message[:160])
                for record in state.evidence[-12:]
                if record.action.target_id == candidate.candidate_id
            ],
        }
        summaries.append((candidate.candidate_id, str(details)))
    for item in state.frontier.values():
        summaries.append((item.frontier_id, f"frontier entity={item.entity_id} depth={item.depth}"))
    for identifier in sorted(state.active_entity_ids):
        entity = state.entities[identifier]
        recent = [
            (record.action.operation.value, record.status, record.message[:160])
            for record in state.evidence[-12:]
            if record.action.target_id == identifier
        ]
        summaries.append(
            (identifier, f"entity {entity.name}; type={entity.entity_type}; recent calls={recent}")
        )
    for record in reversed(state.evidence):
        passages = []
        for anchor in record.anchors:
            if anchor.document_id and record.status == "success":
                source = state.sources[anchor.source_id]
                passages.append(
                    f"{anchor.source_id}[{anchor.start}:{anchor.end}]: {source.text[anchor.start : anchor.end][:700]}"
                )
            elif anchor.image_id:
                passages.append(f"image={anchor.image_id}; source={anchor.source_id}")
        summaries.append(
            (
                record.evidence_id,
                f"{record.action.operation.value} {record.status}; "
                f"target={record.action.target_id}; cost={record.cost}; "
                f"message={record.message[:256]}; evidence={' '.join(passages)}",
            )
        )
    summary = str(
        {
            "budget": state.budget,
            "max_depth": state.max_depth,
            "depth": state.depth,
            "accepted": len(state.graph.assertions),
            "unresolved": len(unresolved),
            "resolved": Counter(
                c.status.value for c in state.candidates.values() if not c.unresolved
            ),
            "failures": Counter(e.status for e in state.evidence if e.status != "success"),
            "provenance": [
                a.source_id
                for assertion in state.graph.assertions.values()
                for a in assertion.anchors
            ][:8],
        }
    )
    return build_context(
        summary,
        summaries,
        [(action.operation.value, action.target_id) for action in actions],
        tokenizer,
        operation_names=tuple(operation.value for operation in Operation),
        max_tokens=max_tokens,
        state_hash=state.hash(),
        evidence_hash=environment.evidence_hash(),
        specialist_signature=stable_hash(environment.specialist_versions),
    )


class FixedRoutePolicy:
    """Fixed retrieval, generation, verification and integration baseline."""

    def __init__(self, abstention_threshold=0.65):
        self.abstention_threshold = abstention_threshold

    def select(self, environment):
        actions = environment.available_actions()
        state = environment.state
        for operation in (Operation.RETRIEVE_TEXT, Operation.RETRIEVE_IMAGE):
            attempted = {
                e.action.target_id for e in state.evidence if e.action.operation is operation
            }
            selected = next(
                (a for a in actions if a.operation is operation and a.target_id not in attempted),
                None,
            )
            if selected:
                return selected
        for operation in (Operation.GENERATE_TEXT, Operation.GENERATE_VISUAL, Operation.VERIFY):
            selected = next((action for action in actions if action.operation is operation), None)
            if selected:
                return selected
        for action in actions:
            verification = state.verifications.get(action.target_id)
            if (
                action.operation is Operation.ACCEPT
                and verification is not None
                and environment.reward_config.proxy(verification) >= self.abstention_threshold
            ):
                return action
        expand = next((action for action in actions if action.operation is Operation.EXPAND), None)
        return expand or Action(Operation.STOP)


def collect_episode(
    environment,
    policy,
    tokenizer,
    *,
    training=False,
    max_tokens=2048,
    deterministic=False,
    max_steps=None,
):
    """Collect a complete episode or a bounded on-policy segment without hidden retries."""
    if training and environment.partition != "ppo":
        raise ValueError("PPO collection requires the PPO training partition")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    steps, trace = [], []
    bound = 1 + math.ceil(environment.state.initial_budget / 0.02)
    while not environment.state.done and len(trace) < (max_steps or bound):
        context = observation_for(environment, tokenizer, max_tokens=max_tokens)
        if hasattr(policy, "select"):
            action = policy.select(environment)
            log_probability = value = operation_index = target_index = None
        else:
            import torch

            policy.eval()
            with torch.no_grad():
                distribution, estimated_value = policy.distribution(context)
                operation_index, target_index = distribution.sample(deterministic=deterministic)
                log_probability = float(distribution.log_prob(operation_index, target_index).item())
                value = float(estimated_value.item())
            action = Action(
                Operation(context.operation_names[operation_index]),
                context.object_ids[target_index],
            )
        transition = environment.step(action)
        if transition.old_state_hash != context.state_hash:
            raise ValueError("Environment snapshot changed during action selection")
        row = {
            "transition": transition.to_dict(),
            "context": context.to_dict(),
            "context_digest": context.digest(),
            "operation_index": operation_index,
            "target_index": target_index,
            "log_probability": log_probability,
            "value": value,
        }
        trace.append(row)
        if training:
            from .training.ppo import RolloutStep

            steps.append(
                RolloutStep(
                    context,
                    operation_index,
                    target_index,
                    log_probability,
                    value,
                    transition.reward,
                    transition.done,
                    transition.new_state_hash,
                    context.digest(),
                    policy_revision=policy.policy_revision,
                    temperature=policy.temperature,
                    policy_identity=policy.policy_identity,
                )
            )
    if max_steps is None and not environment.state.done:
        raise RuntimeError("Episode exceeded its positive-cost action bound")
    if training and environment.state.done:
        from .training.ppo import finalize_episode

        steps = finalize_episode(steps, environment.terminal_reward())
    return steps, trace


def fit_action_costs(rows, *, token_weight, latency_weight, retrieval_weight):
    """Fit validation-average costs using explicitly selected measurement-unit weights."""
    weights = (token_weight, latency_weight, retrieval_weight)
    if any(not math.isfinite(value) or value < 0 for value in weights) or not any(weights):
        raise ValueError("Cost weights must be finite, nonnegative and not all zero")
    samples = {operation.value: [] for operation in Operation if operation is not Operation.STOP}
    for row in rows:
        if row.get("partition") != "validation":
            raise ValueError("Action costs must be estimated from validation calls")
        operation = row.get("operation")
        if operation == "stop":
            continue
        if operation not in samples:
            raise ValueError("Unknown cost observation operation")
        values = [
            row.get(name)
            for name in ("prompt_tokens", "completion_tokens", "latency_seconds", "retrieval_calls")
        ]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise ValueError(
                "Cost observations require nonnegative measured tokens, latency and retrieval calls"
            )
        prompt, completion, latency, calls = values
        samples[operation].append(
            token_weight * (prompt + completion)
            + latency_weight * latency
            + retrieval_weight * calls
        )
    missing = [operation for operation, values in samples.items() if not values]
    if missing:
        raise ValueError(f"Missing validation measurements for: {', '.join(missing)}")
    averages = {operation: sum(values) / len(values) for operation, values in samples.items()}
    largest = max(averages.values())
    if largest <= 0:
        raise ValueError("At least one measured action cost must be positive")
    return {
        **{operation: max(0.02, value / largest) for operation, value in averages.items()},
        "stop": 0.0,
    }
