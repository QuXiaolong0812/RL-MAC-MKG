"""Validated records and deterministic, portable serialization."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import types
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints


def primitive(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: primitive(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(k.value if isinstance(k, Enum) else k): primitive(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [primitive(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((primitive(v) for v in value), key=canonical_json)
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        primitive(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _decode(annotation: Any, value: Any) -> Any:
    if value is None:
        return None
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        for typ in args:
            if typ is not type(None):
                try:
                    return _decode(typ, value)
                except (ValueError, TypeError):
                    continue
        raise ValueError("Value does not match record annotation")
    if origin in (tuple, list, set, frozenset):
        if not isinstance(value, (list, tuple, set, frozenset)):
            raise ValueError("Record sequence must be an array")
        item_type = args[0] if args else Any
        values = [_decode(item_type, item) for item in value]
        return origin(values)
    if origin is dict:
        return {_decode(args[0], k): _decode(args[1], v) for k, v in value.items()}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return annotation.from_dict(value)
    return value


class Record:
    def to_dict(self) -> dict[str, Any]:
        return primitive(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        fields = {f.name for f in dataclasses.fields(cls) if f.init}
        extra = set(value) - fields
        if extra:
            raise ValueError(f"Unknown {cls.__name__} fields: {sorted(extra)}")
        annotations = get_type_hints(cls)
        return cls(**{key: _decode(annotations[key], item) for key, item in value.items()})


def _identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def _tuple(value: Any, item_type: type, field: str) -> tuple:
    if not isinstance(value, (tuple, list)) or any(not isinstance(v, item_type) for v in value):
        raise ValueError(f"{field} must be an array of {item_type.__name__} values")
    return tuple(value)


def _boolean(value: Any, field: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")


def _usage(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or any(
        not isinstance(k, str)
        or isinstance(v, bool)
        or not isinstance(v, (float, int))
        or not math.isfinite(v)
        or v < 0
        for k, v in value.items()
    ):
        raise ValueError("Usage measurements must be finite and nonnegative")
    return dict(value)


def _score(value: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    if not 0 <= value <= 1:
        raise ValueError(f"{field} must lie in [0, 1]")


def _hash(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("content_hash must be a lowercase SHA-256 hash")


def _time(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("retrieved_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("retrieved_at must include a timezone")


class Operation(str, Enum):
    RETRIEVE_TEXT = "retrieve_text"
    RETRIEVE_IMAGE = "retrieve_image"
    GENERATE_TEXT = "generate_text"
    GENERATE_VISUAL = "generate_visual"
    VERIFY = "verify"
    REVISE = "revise"
    ACCEPT = "accept"
    REJECT = "reject"
    EXPAND = "expand"
    STOP = "stop"


@dataclasses.dataclass(frozen=True)
class Action(Record):
    operation: Operation
    target_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "operation", Operation(self.operation))
        if self.operation is Operation.STOP:
            if self.target_id is not None:
                raise ValueError("stop requires a null target")
        else:
            _identifier(self.target_id, "target_id")


@dataclasses.dataclass(frozen=True)
class SourceRecord(Record):
    source_id: str
    entity_ids: tuple[str, ...]
    url: str
    retrieved_at: str
    license: str
    content_hash: str
    kind: str
    text: str | None = None
    path: str | None = None
    document_id: str | None = None
    image_id: str | None = None
    partition: str | None = None
    eligible: bool = True
    duplicate_group: str | None = None
    width: int | None = None
    height: int | None = None
    safety_status: str = "unchecked"

    def __post_init__(self):
        for key in ("source_id", "url", "license"):
            _identifier(getattr(self, key), key)
        _time(self.retrieved_at)
        _hash(self.content_hash)
        object.__setattr__(self, "entity_ids", _tuple(self.entity_ids, str, "entity_ids"))
        _boolean(self.eligible, "eligible")
        for entity_id in self.entity_ids:
            _identifier(entity_id, "entity_id")
        if len(self.entity_ids) != len(set(self.entity_ids)):
            raise ValueError("Duplicate entity_ids in source")
        if self.kind not in ("text", "image"):
            raise ValueError("Source kind must be text or image")
        if self.kind == "text":
            _identifier(self.document_id, "document_id")
            if not isinstance(self.text, str):
                raise ValueError("Text sources require original UTF-8 text")
            if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.content_hash:
                raise ValueError("Source text content hash mismatch")
            if self.image_id is not None:
                raise ValueError("Text source cannot have image_id")
        else:
            _identifier(self.image_id, "image_id")
            if self.text is not None or self.document_id is not None:
                raise ValueError("Image source cannot have document text")
            for value in (self.width, self.height):
                if value is not None and (type(value) is not int or value <= 0):
                    raise ValueError("Image dimensions must be positive integers")


@dataclasses.dataclass(frozen=True)
class SourceAnchor(Record):
    source_id: str
    url: str
    retrieved_at: str
    content_hash: str
    document_id: str | None = None
    start: int | None = None
    end: int | None = None
    image_id: str | None = None
    quote: str | None = None
    region: tuple[float, ...] | None = None

    def __post_init__(self):
        _identifier(self.source_id, "source_id")
        _identifier(self.url, "url")
        _time(self.retrieved_at)
        _hash(self.content_hash)
        if (self.document_id is None) == (self.image_id is None):
            raise ValueError("Anchor requires exactly one document_id or image_id")
        if self.document_id is not None:
            if (
                type(self.start) is not int
                or type(self.end) is not int
                or self.start < 0
                or self.end <= self.start
            ):
                raise ValueError("Text anchor requires a nonempty nonnegative character span")
            if self.region is not None:
                raise ValueError("Text anchor cannot have image region")
        elif self.start is not None or self.end is not None or self.quote is not None:
            raise ValueError("Image anchor cannot have text span")
        if self.region is not None:
            object.__setattr__(self, "region", tuple(self.region))
            if len(self.region) != 4:
                raise ValueError("Image region must have four normalized coordinates")
            for coord in self.region:
                _score(coord, "region coordinate")
            x0, y0, x1, y1 = self.region
            if x0 >= x1 or y0 >= y1:
                raise ValueError("Image region must have positive area")

    def validate(self, sources: Mapping[str, SourceRecord]) -> str | None:
        if self.source_id not in sources:
            raise ValueError("Anchor source is not in the allowed provenance catalog")
        source = sources[self.source_id]
        if not source.eligible:
            raise ValueError("Anchor source is not eligible provenance")
        for key in ("url", "retrieved_at", "content_hash", "document_id", "image_id"):
            if getattr(self, key) != getattr(source, key):
                raise ValueError(f"Anchor {key} mismatch")
        if source.kind == "text":
            if self.end > len(source.text):
                raise ValueError("Anchor character span extends beyond original text")
            text = source.text[self.start : self.end]
            if self.quote is not None and text != self.quote:
                raise ValueError("Anchor quote does not match original character span")
            return text
        if source.path is not None:
            if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != source.content_hash:
                raise ValueError("Image source content hash mismatch")
        return None


@dataclasses.dataclass(frozen=True)
class Entity(Record):
    entity_id: str
    name: str
    entity_type: str = "entity"
    aliases: tuple[str, ...] = ()

    def __post_init__(self):
        for key in ("entity_id", "name", "entity_type"):
            _identifier(getattr(self, key), key)
        object.__setattr__(self, "aliases", _tuple(self.aliases, str, "aliases"))
        for alias in self.aliases:
            _identifier(alias, "alias")


class CandidateStatus(str, Enum):
    PROPOSED = "proposed"
    UNRESOLVED = "proposed"
    EVIDENCED = "evidenced"
    VERIFIED = "verified"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REVISED = "revised"


@dataclasses.dataclass(frozen=True)
class Candidate(Record):
    candidate_id: str
    head: str
    predicate: str
    tail: str
    anchors: tuple[SourceAnchor, ...]
    version: int = 1
    status: CandidateStatus = CandidateStatus.PROPOSED
    modality: str = "text"
    parse_valid: bool = True
    generation_confidence: float | None = None
    created_step: int = 0
    disposition: str | None = None

    def __post_init__(self):
        for key in ("candidate_id", "head", "predicate", "tail"):
            _identifier(getattr(self, key), key)
        object.__setattr__(self, "anchors", _tuple(self.anchors, SourceAnchor, "anchors"))
        object.__setattr__(self, "status", CandidateStatus(self.status))
        _boolean(self.parse_valid, "parse_valid")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Candidate version must be a positive integer")
        if self.modality not in ("text", "visual", "multimodal"):
            raise ValueError("Unknown candidate modality")
        if self.generation_confidence is not None:
            _score(self.generation_confidence, "generation_confidence")
        if type(self.created_step) is not int or self.created_step < 0:
            raise ValueError("created_step must be a nonnegative integer")

    @property
    def unresolved(self) -> bool:
        return self.status in (
            CandidateStatus.PROPOSED,
            CandidateStatus.EVIDENCED,
            CandidateStatus.VERIFIED,
        )


@dataclasses.dataclass(frozen=True)
class Verification(Record):
    candidate_id: str
    candidate_version: int
    factual_support: float
    visual_support: float | None = None
    uncertainty: float = 0.0
    hard_valid: bool = True
    issues: tuple[str, ...] = ()
    specialist_version: str = "unspecified"

    def __post_init__(self):
        _identifier(self.candidate_id, "candidate_id")
        if type(self.candidate_version) is not int or self.candidate_version < 1:
            raise ValueError("Verification candidate_version must be positive")
        _score(self.factual_support, "factual_support")
        _score(self.uncertainty, "uncertainty")
        _boolean(self.hard_valid, "hard_valid")
        if self.visual_support is not None:
            _score(self.visual_support, "visual_support")
        object.__setattr__(self, "issues", _tuple(self.issues, str, "issues"))

    @property
    def support(self) -> float:
        factual_weight = 1.0 if self.visual_support is None else 0.55
        return (
            factual_weight * self.factual_support
            + (0 if self.visual_support is None else 0.45 * self.visual_support)
            - 0.20 * self.uncertainty
        )


@dataclasses.dataclass(frozen=True)
class Conflict(Record):
    conflict_id: str
    head: str
    predicate: str
    values: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    assertion_ids: tuple[str, ...] = ()
    resolved: bool = False
    reason: str = "single-valued predicate disagreement"

    def __post_init__(self):
        for key in ("conflict_id", "head", "predicate", "reason"):
            _identifier(getattr(self, key), key)
        for key in ("values", "candidate_ids", "assertion_ids"):
            object.__setattr__(self, key, _tuple(getattr(self, key), str, key))
        _boolean(self.resolved, "resolved")
        if len(set(self.values)) < 2:
            raise ValueError("Conflict requires at least two different values")


@dataclasses.dataclass(frozen=True)
class FrontierItem(Record):
    frontier_id: str
    entity_id: str
    depth: int
    parent_id: str | None = None

    def __post_init__(self):
        _identifier(self.frontier_id, "frontier_id")
        _identifier(self.entity_id, "entity_id")
        if type(self.depth) is not int or self.depth < 0:
            raise ValueError("Frontier depth must be nonnegative")


@dataclasses.dataclass(frozen=True)
class GraphAssertion(Record):
    assertion_id: str
    head: str
    predicate: str
    tail: str
    anchors: tuple[SourceAnchor, ...]
    candidate_id: str
    candidate_version: int
    support: float = 0.0

    def __post_init__(self):
        for key in ("assertion_id", "head", "predicate", "tail", "candidate_id"):
            _identifier(getattr(self, key), key)
        object.__setattr__(self, "anchors", _tuple(self.anchors, SourceAnchor, "anchors"))
        if not self.anchors:
            raise ValueError("Graph assertion requires provenance anchors")
        if type(self.candidate_version) is not int or self.candidate_version < 1:
            raise ValueError("Assertion candidate_version must be positive")
        _score(self.support, "support")

    @property
    def triple(self) -> tuple[str, str, str]:
        return self.head, self.predicate, self.tail


CALL_STATUSES = frozenset(
    ("success", "empty", "error", "timeout", "invalid", "ambiguous", "unsafe", "parse_error")
)


@dataclasses.dataclass(frozen=True)
class EvidenceRecord(Record):
    evidence_id: str
    action: Action
    status: str
    cost: float
    anchors: tuple[SourceAnchor, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    message: str = ""
    usage: dict[str, float] = dataclasses.field(default_factory=dict)
    request_id: str = ""
    specialist_version: str = "unspecified"
    fingerprint: str = ""
    verification: Verification | None = None

    def __post_init__(self):
        _identifier(self.evidence_id, "evidence_id")
        if not isinstance(self.action, Action):
            raise ValueError("Evidence action must be an Action")
        if self.status not in CALL_STATUSES:
            raise ValueError("Unknown evidence status")
        if not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError("Evidence cost must be finite and nonnegative")
        object.__setattr__(self, "anchors", _tuple(self.anchors, SourceAnchor, "anchors"))
        object.__setattr__(self, "candidate_ids", _tuple(self.candidate_ids, str, "candidate_ids"))
        object.__setattr__(self, "usage", _usage(self.usage))
        if self.verification is not None and not isinstance(self.verification, Verification):
            raise ValueError("verification must be a Verification record")


@dataclasses.dataclass(frozen=True)
class SpecialistResult(Record):
    status: str
    anchors: tuple[SourceAnchor, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    verification: Verification | None = None
    entities: tuple[Entity, ...] = ()
    message: str = ""
    usage: dict[str, float] = dataclasses.field(default_factory=dict)
    request_id: str = ""
    specialist_version: str = "unspecified"

    def __post_init__(self):
        if self.status not in CALL_STATUSES:
            raise ValueError("Unknown specialist status")
        for key, typ in (
            ("anchors", SourceAnchor),
            ("candidates", Candidate),
            ("entities", Entity),
        ):
            object.__setattr__(self, key, _tuple(getattr(self, key), typ, key))
        if self.verification is not None and not isinstance(self.verification, Verification):
            raise ValueError("verification must be a Verification record")
        object.__setattr__(self, "usage", _usage(self.usage))
