"""Deterministic construction transitions and budget-aware hard action masks."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from ..graph import AliasCanonicalizer, GraphStore, Schema, canonical_triple
from ..records import (
    Action,
    Candidate,
    CandidateStatus,
    Conflict,
    Entity,
    EvidenceRecord,
    FrontierItem,
    Operation,
    Record,
    SourceRecord,
    SpecialistResult,
    Verification,
    stable_hash,
)


@dataclass(frozen=True)
class RewardConfig(Record):
    lambda_h: float
    lambda_p: float
    lambda_g: float
    costs: dict[Operation, float]
    lambda_c: float = 0.08
    tau_abs: float = 0.65
    factual_weight: float = 0.55
    visual_weight: float = 0.45
    lambda_u: float = 0.20

    def __post_init__(self):
        for key in (
            "lambda_h",
            "lambda_p",
            "lambda_g",
            "lambda_c",
            "lambda_u",
            "factual_weight",
            "visual_weight",
        ):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{key} must be finite and nonnegative")
        if not math.isfinite(self.tau_abs) or not 0 <= self.tau_abs <= 1:
            raise ValueError("tau_abs must lie in [0, 1]")
        costs = {Operation(key): value for key, value in self.costs.items()}
        if set(costs) != set(Operation):
            raise ValueError("A complete explicit action cost table is required")
        for operation, cost in costs.items():
            if (
                isinstance(cost, bool)
                or not isinstance(cost, (int, float))
                or not math.isfinite(cost)
            ):
                raise ValueError("Action cost must be finite")
            if operation is Operation.STOP and cost != 0:
                raise ValueError("stop cost must equal zero")
            if operation is not Operation.STOP and not 0.02 <= cost <= 1:
                raise ValueError("Non-stop cost must lie in [0.02, 1]")
        object.__setattr__(self, "costs", costs)
        if not math.isclose(self.factual_weight + self.visual_weight, 1.0, abs_tol=1e-9):
            raise ValueError("Factual and visual support weights must sum to one")

    def proxy(self, verification: Verification) -> float:
        factual_weight = 1.0 if verification.visual_support is None else self.factual_weight
        return (
            factual_weight * verification.factual_support
            + (
                0.0
                if verification.visual_support is None
                else self.visual_weight * verification.visual_support
            )
            - self.lambda_u * verification.uncertainty
        )


@dataclass
class ConstructionState(Record):
    entities: dict[str, Entity]
    sources: dict[str, SourceRecord]
    graph: GraphStore
    frontier: dict[str, FrontierItem]
    budget: float
    initial_budget: float
    max_depth: int
    seed_entity_ids: tuple[str, ...]
    candidates: dict[str, Candidate] = field(default_factory=dict)
    candidate_history: dict[str, list[Candidate]] = field(default_factory=dict)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    verifications: dict[str, Verification] = field(default_factory=dict)
    conflicts: dict[str, Conflict] = field(default_factory=dict)
    depth: int = 0
    step_count: int = 0
    done: bool = False
    retrieval_cursors: dict[str, int] = field(default_factory=dict)
    retrieval_exhausted: set[str] = field(default_factory=set)
    call_fingerprints: set[str] = field(default_factory=set)
    expanded_entities: set[str] = field(default_factory=set)
    active_entity_ids: set[str] = field(default_factory=set)
    entity_aliases: dict[str, str] = field(default_factory=dict)

    def hash(self) -> str:
        return stable_hash(self)

    def evidence_hash(self) -> str:
        return stable_hash(self.evidence)


@dataclass(frozen=True)
class Transition(Record):
    old_state_hash: str
    new_state_hash: str
    action: Action
    reward: float
    reward_components: dict[str, float]
    done: bool
    evidence_id: str | None = None


class InvalidAction(ValueError):
    """A rejected action whose feasible attempted execution was recorded and charged."""

    def __init__(self, message: str, transition: Transition | None = None):
        super().__init__(message)
        self.transition = transition


Dispatcher = Callable[[Action, ConstructionState], SpecialistResult]


class Environment:
    """Run one fixed specialist per action and recheck all deterministic constraints.

    Terminal reward is separate from step reward and is available only to PPO training.
    References are private environment data and never enter ConstructionState.
    """

    def __init__(
        self,
        entities: Iterable[Entity],
        sources: Iterable[SourceRecord],
        schema: Schema,
        reward_config: RewardConfig,
        dispatcher: Dispatcher | None = None,
        budget: float = 30,
        max_depth: int = 3,
        specialist_versions: dict[str, str] | None = None,
        partition: str = "inference",
        reference_graph: Iterable[tuple[str, str, str]] | None = None,
        canonicalizer: Callable[[str], str] | None = None,
        seed_entity_ids: Iterable[str] | None = None,
    ):
        if isinstance(budget, bool) or not math.isfinite(budget) or budget < 0:
            raise ValueError("Budget must be finite and nonnegative")
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        if reference_graph is not None and partition != "ppo":
            raise ValueError("Reference graph rewards are allowed only in the PPO partition")
        entities, sources = list(entities), list(sources)
        entity_map = {entity.entity_id: entity for entity in entities}
        source_map = {source.source_id: source for source in sources}
        if len(entity_map) != len(entities) or len(source_map) != len(sources):
            raise ValueError("Duplicate entity or source identifiers")
        if any(e not in entity_map for source in sources for e in source.entity_ids):
            raise ValueError("Source references unknown entity")
        if partition != "inference" and any(s.partition != partition for s in sources):
            raise ValueError("Source partition differs from episode partition")
        default_seeds = sorted({e for source in sources for e in source.entity_ids}) or sorted(
            entity_map
        )
        seeds = tuple(sorted(seed_entity_ids if seed_entity_ids is not None else default_seeds))
        if len(seeds) != len(set(seeds)) or any(e not in entity_map for e in seeds):
            raise ValueError("Invalid seed entity IDs")
        frontier = {f"frontier:{e}:0": FrontierItem(f"frontier:{e}:0", e, 0) for e in seeds}
        self.state = ConstructionState(
            entity_map,
            source_map,
            GraphStore(),
            frontier,
            float(budget),
            float(budget),
            max_depth,
            seeds,
        )
        self.state.active_entity_ids.update(seeds)
        self.schema = schema
        self.reward_config = reward_config
        self.dispatcher = dispatcher
        self.specialist_versions = dict(specialist_versions or {})
        self.partition = partition
        self._reference_graph = None if reference_graph is None else set(reference_graph)
        self._fixed_canonicalizer = canonicalizer or AliasCanonicalizer()
        self.state.entity_aliases = self._build_entity_aliases(self.state.entities)
        self.canonicalizer = self._canonicalize

    def _build_entity_aliases(self, entities: dict[str, Entity]) -> dict[str, str]:
        aliases = {}
        for entity in entities.values():
            for alias in (entity.name, *entity.aliases):
                if alias == entity.entity_id:
                    continue
                if alias in aliases and aliases[alias] != entity.entity_id:
                    raise ValueError("Ambiguous canonical entity name or alias")
                if alias in entities and alias != entity.entity_id:
                    raise ValueError("Entity alias collides with a different entity ID")
                fixed = self._fixed_canonicalizer(alias)
                if fixed in entities and fixed != entity.entity_id:
                    raise ValueError("Entity alias conflicts with the fixed canonicalizer")
                aliases[alias] = entity.entity_id
        return aliases

    def _canonicalize(self, value: str) -> str:
        if value in self.state.entities:
            return value
        if value in self.state.entity_aliases:
            return self.state.entity_aliases[value]
        fixed = self._fixed_canonicalizer(value)
        return self.state.entity_aliases.get(fixed, fixed)

    def evidence_hash(self) -> str:
        return self.state.evidence_hash()

    def _related_evidence(self, target: str) -> list[Any]:
        candidate = self.state.candidates.get(target)
        entity_ids = {self.canonicalizer(candidate.head) if candidate else target}
        target_evidence = next((e for e in self.state.evidence if e.evidence_id == target), None)
        if (
            target_evidence is not None
            and target_evidence.status == "success"
            and target_evidence.action.operation
            in (Operation.RETRIEVE_TEXT, Operation.RETRIEVE_IMAGE)
        ):
            entity_ids.update(
                entity_id
                for anchor in target_evidence.anchors
                if anchor.source_id in self.state.sources
                for entity_id in self.state.sources[anchor.source_id].entity_ids
            )
        related_source_ids = {
            s.source_id
            for s in self.state.sources.values()
            if entity_ids.intersection(s.entity_ids)
        }
        anchors = {
            stable_hash(anchor): anchor
            for evidence in self.state.evidence
            if evidence.status == "success"
            for anchor in evidence.anchors
            if anchor.source_id in related_source_ids
        }
        if candidate:
            anchors.update({stable_hash(a): a for a in candidate.anchors})
        return [anchors[key].to_dict() for key in sorted(anchors)]

    def _fingerprint(self, action: Action) -> str:
        target = action.target_id
        candidate = self.state.candidates.get(target)
        data = {
            "operation": action.operation.value,
            "target": target,
            "version": candidate.version if candidate else None,
            "related_evidence": self._related_evidence(target),
        }
        if action.operation in (Operation.RETRIEVE_TEXT, Operation.RETRIEVE_IMAGE):
            kind = "text" if action.operation is Operation.RETRIEVE_TEXT else "image"
            data["cursor"] = self.state.retrieval_cursors.get(f"{target}:{kind}", 0)
        if action.operation in (Operation.GENERATE_TEXT, Operation.GENERATE_VISUAL):
            data["source_evidence"] = target
        return stable_hash(data)

    def _accept_issues(self, candidate: Candidate) -> list[str]:
        issues = self.state.graph.issues(
            candidate, self.schema, self.state.entities, self.canonicalizer
        )
        if not candidate.anchors:
            issues.append("missing allowed provenance")
        for anchor in candidate.anchors:
            try:
                anchor.validate(self.state.sources)
            except (ValueError, OSError) as exc:
                issues.append(f"invalid provenance: {exc}")
        if candidate.predicate in ("hasImage", "has_image"):
            image_ids = {a.image_id for a in candidate.anchors if a.image_id is not None}
            if candidate.tail not in image_ids:
                issues.append("hasImage requires a matching image anchor")
        verification = self.state.verifications.get(candidate.candidate_id)
        if verification is not None:
            if verification.candidate_version != candidate.version:
                issues.append("stale verification version")
            if not verification.hard_valid:
                issues.append("verification hard rule failed")
        return issues

    def _issues(self, action: Action) -> list[str]:
        if action.operation is Operation.STOP:
            return []
        issues = []
        if self.reward_config.costs[action.operation] > self.state.budget + 1e-12:
            issues.append("insufficient budget")
        operation, target = action.operation, action.target_id
        if operation in (Operation.RETRIEVE_TEXT, Operation.RETRIEVE_IMAGE):
            if target not in self.state.active_entity_ids:
                issues.append("retrieval target must be an active seed or expanded entity")
            kind = "text" if operation is Operation.RETRIEVE_TEXT else "image"
            if f"{target}:{kind}" in self.state.retrieval_exhausted:
                issues.append("retrieval inventory is exhausted")
        elif operation in (Operation.GENERATE_TEXT, Operation.GENERATE_VISUAL):
            evidence = next((e for e in self.state.evidence if e.evidence_id == target), None)
            kind = "text" if operation is Operation.GENERATE_TEXT else "image"
            retrieval_operation = (
                Operation.RETRIEVE_TEXT if kind == "text" else Operation.RETRIEVE_IMAGE
            )
            if (
                evidence is None
                or evidence.status != "success"
                or evidence.action.operation is not retrieval_operation
                or not evidence.anchors
                or not any(
                    a.source_id in self.state.sources
                    and self.state.sources[a.source_id].kind == kind
                    for a in evidence.anchors
                )
            ):
                issues.append("generation target must be compatible successful retrieval evidence")
        elif operation is Operation.EXPAND:
            frontier = self.state.frontier.get(target)
            if frontier is None:
                issues.append("expansion target must be an active frontier item")
            elif frontier.depth >= self.state.max_depth:
                issues.append("maximum expansion depth reached")
        else:
            candidate = self.state.candidates.get(target)
            if candidate is None or not candidate.unresolved:
                issues.append("target must be an unresolved candidate")
            elif operation is Operation.ACCEPT:
                issues.extend(self._accept_issues(candidate))
        if self._fingerprint(action) in self.state.call_fingerprints:
            issues.append("duplicate call without new relevant evidence")
        return issues

    def available_actions(self) -> list[Action]:
        if self.state.done:
            return [Action(Operation.STOP)]
        potential = []
        for operation in (Operation.RETRIEVE_TEXT, Operation.RETRIEVE_IMAGE):
            potential.extend(
                Action(operation, target) for target in sorted(self.state.active_entity_ids)
            )
        for evidence in self.state.evidence:
            if evidence.status != "success":
                continue
            if evidence.action.operation is Operation.RETRIEVE_TEXT:
                potential.append(Action(Operation.GENERATE_TEXT, evidence.evidence_id))
            elif evidence.action.operation is Operation.RETRIEVE_IMAGE:
                potential.append(Action(Operation.GENERATE_VISUAL, evidence.evidence_id))
        for target, candidate in sorted(self.state.candidates.items()):
            if candidate.unresolved:
                potential.extend(
                    Action(operation, target)
                    for operation in (
                        Operation.VERIFY,
                        Operation.REVISE,
                        Operation.ACCEPT,
                        Operation.REJECT,
                    )
                )
        potential.extend(Action(Operation.EXPAND, target) for target in sorted(self.state.frontier))
        return [action for action in potential if not self._issues(action)] + [
            Action(Operation.STOP)
        ]

    def add_candidate(self, candidate: Candidate) -> None:
        if candidate.candidate_id in self.state.candidates:
            raise ValueError("Candidate ID already exists; use revise to create a new version")
        if candidate.version != 1 or not candidate.unresolved:
            raise ValueError("New candidates must start unresolved at version 1")
        status = CandidateStatus.EVIDENCED if candidate.anchors else CandidateStatus.PROPOSED
        self.state.candidates[candidate.candidate_id] = replace(
            candidate, status=status, created_step=self.state.step_count
        )
        self.state.candidate_history[candidate.candidate_id] = []
        self._detect_conflicts(candidate)

    def _detect_conflicts(self, candidate: Candidate) -> None:
        h, p, t = canonical_triple(candidate, self.schema, self.canonicalizer)
        rule = self.schema.predicates.get(p)
        if rule is None or not rule.single_valued:
            return
        competing = [
            c
            for c in self.state.candidates.values()
            if self.canonicalizer(c.head) == h
            and c.predicate == p
            and canonical_triple(c, self.schema, self.canonicalizer)[2] != t
            and c.status is not CandidateStatus.REJECTED
        ]
        assertions = [
            a
            for a in self.state.graph.assertions.values()
            if a.head == h and a.predicate == p and a.tail != t
        ]
        if not competing and not assertions:
            return
        ids = tuple(sorted({candidate.candidate_id, *(c.candidate_id for c in competing)}))
        values = tuple(
            sorted(
                {
                    t,
                    *(canonical_triple(c, self.schema, self.canonicalizer)[2] for c in competing),
                    *(a.tail for a in assertions),
                }
            )
        )
        conflict_id = "conflict:" + stable_hash((h, p, values, ids))[:20]
        self.state.conflicts[conflict_id] = Conflict(
            conflict_id, h, p, values, ids, tuple(sorted(a.assertion_id for a in assertions))
        )

    def _refresh_conflicts(self) -> None:
        for identifier, conflict in self.state.conflicts.items():
            values = {
                canonical_triple(c, self.schema, self.canonicalizer)[2]
                for c in self.state.candidates.values()
                if c.status is not CandidateStatus.REJECTED
                and self.canonicalizer(c.head) == conflict.head
                and c.predicate == conflict.predicate
            }
            values.update(
                a.tail
                for a in self.state.graph.assertions.values()
                if a.head == conflict.head and a.predicate == conflict.predicate
            )
            self.state.conflicts[identifier] = replace(conflict, resolved=len(values) < 2)

    def record_verification(self, verification: Verification) -> None:
        candidate = self.state.candidates.get(verification.candidate_id)
        if candidate is None or not candidate.unresolved:
            raise ValueError("Verification requires an unresolved candidate")
        if verification.candidate_version != candidate.version:
            raise ValueError("Verification version is stale")
        self.state.verifications[candidate.candidate_id] = verification
        self.state.candidates[candidate.candidate_id] = replace(
            candidate, status=CandidateStatus.VERIFIED
        )

    def _quality_components(self) -> dict[str, float]:
        accepted = list(self.state.graph.assertions.values())
        resolved = [c for c in self.state.candidates.values() if not c.unresolved]
        q = sum(max(0.0, min(1.0, a.support)) for a in accepted) / max(1, len(resolved))
        p = sum(bool(a.anchors) for a in accepted) / max(1, len(accepted))
        contradiction_count = sum(
            any(
                other.head == assertion.head
                and other.predicate == assertion.predicate
                and other.tail != assertion.tail
                for other in accepted
            )
            for assertion in accepted
            if self.schema.predicates[assertion.predicate].single_valued
        )
        c = contradiction_count / max(1, len(accepted))
        d = (len(accepted) - len(self.state.graph.triples())) / max(1, len(accepted))
        discovered = {
            f.entity_id for f in self.state.frontier.values()
        } | self.state.expanded_entities
        new = discovered - set(self.state.seed_entity_ids)
        i = min(1.0, len(new) / max(1, len(discovered)))
        return {"Q": q, "P": p, "C": c, "D": d, "I": i}

    def quality(self) -> float:
        q = self._quality_components()
        return 0.35 * q["Q"] + 0.20 * q["P"] - 0.20 * q["C"] - 0.15 * q["D"] + 0.10 * q["I"]

    def _append_evidence(self, action, result, cost, fingerprint) -> EvidenceRecord:
        identifier = f"evidence:{len(self.state.evidence) + 1:06d}"
        if action.operation in (Operation.ACCEPT, Operation.REJECT, Operation.EXPAND):
            result = replace(
                result,
                specialist_version=self.specialist_versions.get(
                    action.operation.value, result.specialist_version
                ),
            )
        record = EvidenceRecord(
            identifier,
            action,
            result.status,
            cost,
            result.anchors,
            tuple(c.candidate_id for c in result.candidates),
            result.message,
            dict(result.usage),
            result.request_id or identifier,
            result.specialist_version,
            fingerprint,
            result.verification,
        )
        self.state.evidence.append(record)
        self.state.call_fingerprints.add(fingerprint)
        return record

    def _dispatch(self, action: Action) -> SpecialistResult:
        expected_version = self.specialist_versions.get(action.operation.value, "unspecified")
        if self.dispatcher is None:
            return SpecialistResult(
                "error",
                message="No specialist dispatcher configured",
                specialist_version=expected_version,
            )
        try:
            result = self.dispatcher(action, copy.deepcopy(self.state))
            if not isinstance(result, SpecialistResult):
                raise ValueError("Dispatcher must return SpecialistResult")
            expected_version = self.specialist_versions.get(action.operation.value)
            if expected_version is not None and expected_version != result.specialist_version:
                return replace(result, status="invalid", message="Specialist version mismatch")
            return result
        except TimeoutError as exc:
            return SpecialistResult(
                "timeout",
                message=str(exc) or "Specialist timed out",
                specialist_version=expected_version,
            )
        except Exception as exc:
            return SpecialistResult(
                "error", message=f"{type(exc).__name__}: {exc}", specialist_version=expected_version
            )

    def _apply_output(self, action: Action, result: SpecialistResult) -> None:
        for anchor in result.anchors:
            anchor.validate(self.state.sources)
        operation, target = action.operation, action.target_id
        if result.entities and operation not in (
            Operation.GENERATE_TEXT,
            Operation.GENERATE_VISUAL,
            Operation.REVISE,
        ):
            raise ValueError("Only generation and revision may propose entities")
        if operation in (Operation.RETRIEVE_TEXT, Operation.RETRIEVE_IMAGE):
            kind = "text" if operation is Operation.RETRIEVE_TEXT else "image"
            if not result.anchors:
                raise ValueError("Empty retrieval cannot be marked successful")
            if any(self.state.sources[a.source_id].kind != kind for a in result.anchors):
                raise ValueError("Retrieved source kind is incompatible with operation")
            if any(
                target not in self.state.sources[a.source_id].entity_ids for a in result.anchors
            ):
                raise ValueError("Retrieved source is not assigned to target entity")
            acquired = [
                anchor
                for evidence in self.state.evidence
                if evidence.status == "success" and evidence.action == action
                for anchor in evidence.anchors
            ]
            if all(
                any(self._contains_anchor(old, anchor) for old in acquired)
                for anchor in result.anchors
            ):
                raise ValueError("Duplicate retrieval returned no new anchored information")
            key = f"{target}:{kind}"
            self.state.retrieval_cursors[key] = self.state.retrieval_cursors.get(key, 0) + 1
        elif operation in (Operation.GENERATE_TEXT, Operation.GENERATE_VISUAL):
            if not result.candidates:
                raise ValueError("Successful generation requires at least one candidate")
            ids = [c.candidate_id for c in result.candidates]
            if len(set(ids)) != len(ids) or any(i in self.state.candidates for i in ids):
                raise ValueError("Duplicate candidate ID in generation output")
            self._incorporate_entities(result)
            for candidate in result.candidates:
                if candidate.version != 1 or not candidate.unresolved:
                    raise ValueError("New candidates must start unresolved at version 1")
                for anchor in candidate.anchors:
                    anchor.validate(self.state.sources)
                self._validate_candidate_output(action, candidate)
            for candidate in result.candidates:
                self.add_candidate(candidate)
        elif operation is Operation.VERIFY:
            if result.verification is None or result.verification.candidate_id != target:
                raise ValueError("Verification result does not match target")
            self.record_verification(result.verification)
        elif operation is Operation.REVISE:
            if len(result.candidates) != 1:
                raise ValueError("Revision requires exactly one replacement candidate")
            old, proposed = self.state.candidates[target], result.candidates[0]
            if proposed.candidate_id != target:
                raise ValueError("Revision candidate ID does not match target")
            self._incorporate_entities(result)
            for anchor in proposed.anchors:
                anchor.validate(self.state.sources)
            self._validate_candidate_output(action, proposed)
            self.state.candidate_history[target].append(
                replace(old, status=CandidateStatus.REVISED, disposition="revised")
            )
            self.state.candidates[target] = replace(
                proposed,
                version=old.version + 1,
                status=CandidateStatus.EVIDENCED if proposed.anchors else CandidateStatus.PROPOSED,
                created_step=self.state.step_count,
                disposition=None,
            )
            self.state.verifications.pop(target, None)
            self._detect_conflicts(self.state.candidates[target])
            self._refresh_conflicts()

    def _incorporate_entities(self, result: SpecialistResult) -> None:
        entity_map = dict(self.state.entities)
        ids = [entity.entity_id for entity in result.entities]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate entity ID in specialist output")
        for entity in result.entities:
            old = entity_map.get(entity.entity_id)
            if old is not None and old != entity:
                raise ValueError("Entity proposal conflicts with existing ID/name/type/aliases")
            fixed_id = self._fixed_canonicalizer(entity.entity_id)
            if fixed_id in entity_map and fixed_id != entity.entity_id:
                raise ValueError("Entity proposal ID conflicts with fixed canonical identity")
            entity_map[entity.entity_id] = entity
        aliases = self._build_entity_aliases(entity_map)

        def canonical(value):
            if value in entity_map:
                return value
            return aliases.get(value, self._fixed_canonicalizer(value))

        endpoints = set()
        for candidate in result.candidates:
            endpoints.add(canonical(candidate.head))
            rule = self.schema.predicates.get(candidate.predicate)
            if rule is not None and rule.tail_is_entity:
                endpoints.add(canonical(candidate.tail))
        if any(entity_id not in endpoints for entity_id in set(ids) - set(self.state.entities)):
            raise ValueError("New entity proposal must participate in an output relation")
        self.state.entities = entity_map
        self.state.entity_aliases = aliases

    def _validate_candidate_output(self, action: Action, candidate: Candidate) -> None:
        head, predicate, tail = canonical_triple(candidate, self.schema, self.canonicalizer)
        if head not in self.state.active_entity_ids:
            raise ValueError("Candidate head must be an active seed or expanded entity")
        rule = self.schema.predicates.get(predicate)
        if rule is None:
            raise ValueError("Generated predicate is not allowed by schema")
        if (
            head not in self.state.entities
            or self.state.entities[head].entity_type not in rule.head_types
        ):
            raise ValueError("Generated head endpoint fails schema")
        if rule.tail_is_entity and (
            tail not in self.state.entities
            or self.state.entities[tail].entity_type not in rule.tail_types
        ):
            raise ValueError("Generated tail endpoint fails schema")
        supports = []
        if action.operation is Operation.REVISE:
            old = self.state.candidates[action.target_id]
            if head != self.canonicalizer(old.head):
                raise ValueError("Revision must preserve the candidate head entity")
            supports.extend(old.anchors)
        else:
            origin = next(e for e in self.state.evidence if e.evidence_id == action.target_id)
            if origin.action.target_id != head:
                raise ValueError("Generated candidate head differs from retrieval target")
        supports.extend(
            anchor
            for evidence in self.state.evidence
            if evidence.status == "success"
            and evidence.action.operation in (Operation.RETRIEVE_TEXT, Operation.RETRIEVE_IMAGE)
            and evidence.action.target_id == head
            for anchor in evidence.anchors
        )
        for anchor in candidate.anchors:
            if head not in self.state.sources[anchor.source_id].entity_ids:
                raise ValueError("Candidate source is not associated with its active head")
            if not any(self._contains_anchor(support, anchor) for support in supports):
                raise ValueError("Candidate anchor was not acquired for its active head")

    @staticmethod
    def _contains_anchor(support, anchor) -> bool:
        if (support.source_id, support.content_hash, support.document_id, support.image_id) != (
            anchor.source_id,
            anchor.content_hash,
            anchor.document_id,
            anchor.image_id,
        ):
            return False
        if anchor.document_id is not None:
            return support.start <= anchor.start < anchor.end <= support.end
        if support.region is None:
            return True
        if anchor.region is None:
            return False
        return (
            support.region[0] <= anchor.region[0] < anchor.region[2] <= support.region[2]
            and support.region[1] <= anchor.region[1] < anchor.region[3] <= support.region[3]
        )

    def _expand(self, target: str) -> None:
        item = self.state.frontier.pop(target)
        next_depth = item.depth + 1
        self.state.depth = max(self.state.depth, next_depth)
        self.state.expanded_entities.add(item.entity_id)
        neighbors = {
            a.tail
            for a in self.state.graph.assertions.values()
            if a.head == item.entity_id and self.schema.predicates[a.predicate].tail_is_entity
        }
        neighbors.update(
            a.head
            for a in self.state.graph.assertions.values()
            if a.tail == item.entity_id and self.schema.predicates[a.predicate].tail_is_entity
        )
        known = {f.entity_id for f in self.state.frontier.values()} | self.state.expanded_entities
        for entity_id in sorted(neighbors - known):
            if entity_id in self.state.entities:
                identifier = f"frontier:{entity_id}:{next_depth}"
                self.state.frontier[identifier] = FrontierItem(
                    identifier, entity_id, next_depth, item.frontier_id
                )
                self.state.active_entity_ids.add(entity_id)

    def _pending_frontier_work(self) -> bool:
        for entity_id in self.state.active_entity_ids:
            for kind, operation in (
                ("text", Operation.RETRIEVE_TEXT),
                ("image", Operation.RETRIEVE_IMAGE),
            ):
                available = [
                    s
                    for s in self.state.sources.values()
                    if entity_id in s.entity_ids and s.kind == kind and s.eligible
                ]
                # Cursors index specialist-produced chunks/images, not source records.
                if available and not self._issues(Action(operation, entity_id)):
                    return True
        for evidence in self.state.evidence:
            if evidence.status == "success" and evidence.action.operation in (
                Operation.RETRIEVE_TEXT,
                Operation.RETRIEVE_IMAGE,
            ):
                operation = (
                    Operation.GENERATE_TEXT
                    if evidence.action.operation is Operation.RETRIEVE_TEXT
                    else Operation.GENERATE_VISUAL
                )
                if not self._issues(Action(operation, evidence.evidence_id)):
                    return True
        return False

    def _update_termination(self) -> None:
        unresolved = any(candidate.unresolved for candidate in self.state.candidates.values())
        if not self.state.frontier and not unresolved:
            self.state.done = True
        if (
            self.state.depth >= self.state.max_depth
            and not unresolved
            and not self._pending_frontier_work()
        ):
            self.state.done = True
        if not any(a.operation is not Operation.STOP for a in self.available_actions()):
            self.state.done = True

    def _finish(
        self,
        old_hash: str,
        old_quality: float,
        action: Action,
        hard: float,
        proxy: float,
        cost: float,
        evidence_id: str | None,
    ) -> Transition:
        delta = max(-1.0, min(1.0, self.quality() - old_quality))
        rc = self.reward_config
        reward = rc.lambda_h * hard + rc.lambda_p * proxy + rc.lambda_g * delta - rc.lambda_c * cost
        components = {"hard": hard, "proxy": proxy, "delta_quality": delta, "cost": cost}
        return Transition(
            old_hash, self.state.hash(), action, reward, components, self.state.done, evidence_id
        )

    def step(self, action: Action) -> Transition:
        if not isinstance(action, Action):
            raise TypeError("step requires an Action")
        if self.state.done:
            raise InvalidAction("Episode has already terminated")
        old_hash, old_quality = self.state.hash(), self.quality()
        if action.operation is Operation.STOP:
            self.state.done = True
            self.state.step_count += 1
            return self._finish(old_hash, old_quality, action, 0, 0, 0, None)
        issues, fingerprint = self._issues(action), self._fingerprint(action)
        cost = self.reward_config.costs[action.operation]
        if issues:
            # Unaffordable requests cannot spend funds that do not exist.
            charged = cost if cost <= self.state.budget + 1e-12 else 0.0
            self.state.budget = max(0.0, self.state.budget - charged)
            record = self._append_evidence(
                action, SpecialistResult("invalid", message="; ".join(issues)), charged, fingerprint
            )
            self.state.step_count += 1
            self._update_termination()
            transition = self._finish(
                old_hash, old_quality, action, -1, 0, charged, record.evidence_id
            )
            raise InvalidAction("; ".join(issues), transition)
        self.state.budget = max(0.0, self.state.budget - cost)
        hard, proxy = 0.0, 0.0
        operation, target = action.operation, action.target_id
        if operation is Operation.ACCEPT:
            # Revalidate immediately before graph mutation, independently of policy masks.
            candidate = self.state.candidates[target]
            issues = self._accept_issues(candidate)
            if issues:
                result, hard = SpecialistResult("invalid", message="; ".join(issues)), -1.0
            else:
                verification = self.state.verifications.get(target)
                proxy = self.reward_config.proxy(verification) if verification is not None else 0.0
                self.state.graph.add(candidate, max(0.0, proxy), self.canonicalizer, self.schema)
                self.state.candidates[target] = replace(
                    candidate, status=CandidateStatus.ACCEPTED, disposition="accepted"
                )
                result, hard = SpecialistResult("success", anchors=candidate.anchors), 1.0
        elif operation is Operation.REJECT:
            candidate = self.state.candidates[target]
            self.state.candidates[target] = replace(
                candidate, status=CandidateStatus.REJECTED, disposition="rejected"
            )
            self._refresh_conflicts()
            result = SpecialistResult("success")
        elif operation is Operation.EXPAND:
            self._expand(target)
            result = SpecialistResult("success")
        else:
            result = self._dispatch(action)
            if (
                operation in (Operation.RETRIEVE_TEXT, Operation.RETRIEVE_IMAGE)
                and result.status == "empty"
            ):
                kind = "text" if operation is Operation.RETRIEVE_TEXT else "image"
                self.state.retrieval_exhausted.add(f"{target}:{kind}")
            if result.status == "success":
                snapshot = copy.deepcopy(self.state)
                try:
                    self._apply_output(action, result)
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    self.state = snapshot
                    result = replace(result, status="invalid", message=str(exc))
            if result.status != "success":
                hard = -1.0
        record = self._append_evidence(action, result, cost, fingerprint)
        self.state.step_count += 1
        self._update_termination()
        return self._finish(old_hash, old_quality, action, hard, proxy, cost, record.evidence_id)

    def terminal_reward(self) -> float:
        if not self.state.done:
            raise ValueError("Terminal reward requires a terminated episode")
        if self.partition != "ppo" or self._reference_graph is None:
            return 0.0
        predicted, reference = self.state.graph.triples(), self._reference_graph
        denominator = len(predicted) + len(reference)
        f1 = 2 * len(predicted & reference) / denominator if denominator else 1.0
        components = self._quality_components()
        return 0.60 * f1 - 0.10 * components["C"] - 0.10 * components["D"]


__all__ = [
    "ConstructionState",
    "Dispatcher",
    "Environment",
    "InvalidAction",
    "RewardConfig",
    "Transition",
]
