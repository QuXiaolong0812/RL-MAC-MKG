"""Deterministic schema validation, exact aliases, and canonical graph storage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..records import Candidate, Entity, GraphAssertion, Record, stable_hash


class AliasCanonicalizer:
    """Resolve explicitly supplied aliases; no fuzzy string equivalence is assumed."""

    def __init__(self, aliases: dict[str, str] | None = None):
        self.aliases = dict(aliases or {})
        for alias in self.aliases:
            seen, current = set(), alias
            while current in self.aliases:
                if current in seen:
                    raise ValueError("Cyclic canonical aliases")
                seen.add(current)
                current = self.aliases[current]

    def __call__(self, value: str) -> str:
        while value in self.aliases:
            value = self.aliases[value]
        return value


@dataclass(frozen=True)
class PredicateRule(Record):
    head_types: tuple[str, ...]
    tail_types: tuple[str, ...] = ()
    single_valued: bool = False
    tail_is_entity: bool = True

    def __post_init__(self):
        object.__setattr__(self, "head_types", tuple(self.head_types))
        object.__setattr__(self, "tail_types", tuple(self.tail_types))
        if not self.head_types or (self.tail_is_entity and not self.tail_types):
            raise ValueError("Entity predicates require explicit allowed head and tail types")


@dataclass(frozen=True)
class Schema(Record):
    predicates: dict[str, PredicateRule]

    def __post_init__(self):
        if not self.predicates or any(not p for p in self.predicates):
            raise ValueError("Schema must define allowed predicates")
        if any(not isinstance(v, PredicateRule) for v in self.predicates.values()):
            raise ValueError("Schema predicates must contain PredicateRule records")


def canonical_triple(
    candidate: Candidate, schema: Schema, canonicalizer: Callable[[str], str]
) -> tuple[str, str, str]:
    """Canonicalize entity positions while preserving exact predicates and literal values."""
    rule = schema.predicates.get(candidate.predicate)
    tail = (
        canonicalizer(candidate.tail)
        if rule is not None and rule.tail_is_entity
        else candidate.tail
    )
    return canonicalizer(candidate.head), candidate.predicate, tail


@dataclass
class GraphStore(Record):
    assertions: dict[str, GraphAssertion] = field(default_factory=dict)

    def triples(self) -> set[tuple[str, str, str]]:
        return {assertion.triple for assertion in self.assertions.values()}

    def issues(
        self,
        candidate: Candidate,
        schema: Schema,
        entities: dict[str, Entity],
        canonicalizer: Callable[[str], str],
    ) -> list[str]:
        h, p, t = canonical_triple(candidate, schema, canonicalizer)
        issues = []
        if not candidate.parse_valid:
            issues.append("invalid parse")
        rule = schema.predicates.get(p)
        if rule is None:
            issues.append("predicate is not allowed by schema")
            return issues
        if h not in entities or entities[h].entity_type not in rule.head_types:
            issues.append("head type fails schema")
        if rule.tail_is_entity and (
            t not in entities or entities[t].entity_type not in rule.tail_types
        ):
            issues.append("tail type fails schema")
        if (h, p, t) in self.triples():
            issues.append("duplicate assertion")
        if rule.single_valued and any(
            a.head == h and a.predicate == p and a.tail != t for a in self.assertions.values()
        ):
            issues.append("single-valued predicate contradiction")
        return issues

    def add(
        self,
        candidate: Candidate,
        support: float,
        canonicalizer: Callable[[str], str],
        schema: Schema,
    ) -> GraphAssertion:
        h, p, t = canonical_triple(candidate, schema, canonicalizer)
        identifier = "assertion:" + stable_hash((h, p, t))[:20]
        if identifier in self.assertions:
            raise ValueError("Duplicate assertion")
        assertion = GraphAssertion(
            identifier,
            h,
            p,
            t,
            candidate.anchors,
            candidate.candidate_id,
            candidate.version,
            support,
        )
        self.assertions[identifier] = assertion
        return assertion


__all__ = ["AliasCanonicalizer", "GraphStore", "PredicateRule", "Schema", "canonical_triple"]
