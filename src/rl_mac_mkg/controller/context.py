"""Bounded, reproducible controller observations and target visibility."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DecisionContext:
    input_ids: tuple[int, ...]
    object_ids: tuple[str | None, ...]
    object_positions: tuple[int, ...]
    mask: tuple[tuple[bool, ...], ...]
    operation_names: tuple[str, ...]
    state_hash: str
    evidence_hash: str
    specialist_signature: str

    def __post_init__(self):
        if not self.input_ids or min(self.input_ids) < 0:
            raise ValueError("Context requires nonnegative token IDs")
        if not self.object_ids or self.object_ids[0] is not None:
            raise ValueError("The first target must be null")
        if len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("Object IDs must be unique")
        if len(self.object_positions) != len(self.object_ids):
            raise ValueError("Each object requires a token position")
        if any(not 0 <= position < len(self.input_ids) for position in self.object_positions):
            raise ValueError("Object position outside context")
        if len(self.mask) != len(self.operation_names) or len(set(self.operation_names)) != len(
            self.mask
        ):
            raise ValueError("Operation names and mask rows must agree")
        if any(len(row) != len(self.object_ids) for row in self.mask):
            raise ValueError("Mask width must match visible objects")
        if any(type(entry) is not bool for row in self.mask for entry in row):
            raise ValueError("Action mask must contain booleans")
        if "stop" not in self.operation_names:
            raise ValueError("Stop must be present")
        stop_mask = self.mask[self.operation_names.index("stop")]
        if not stop_mask[0] or any(stop_mask[1:]):
            raise ValueError("Stop must have only a null pointer")
        if any(row[0] for i, row in enumerate(self.mask) if self.operation_names[i] != "stop"):
            raise ValueError("Only stop may use a null pointer")
        if not self.state_hash or not self.evidence_hash or not self.specialist_signature:
            raise ValueError("Context must identify its state, evidence and specialists")

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]):
        return cls(
            **{
                **value,
                "input_ids": tuple(value["input_ids"]),
                "object_ids": tuple(value["object_ids"]),
                "object_positions": tuple(value["object_positions"]),
                "operation_names": tuple(value["operation_names"]),
                "mask": tuple(tuple(row) for row in value["mask"]),
            }
        )


def build_context(
    summary: str,
    objects: Sequence[tuple[str, str]],
    feasible_actions: Sequence[tuple[str, str | None]],
    tokenizer,
    *,
    operation_names: Sequence[str],
    max_tokens: int = 2048,
    state_hash: str,
    evidence_hash: str,
    specialist_signature: str,
) -> DecisionContext:
    """Pack visible objects before building their action mask, without silent truncation."""
    if max_tokens < 64:
        raise ValueError("max_tokens must be at least 64")
    if len({identifier for identifier, _ in objects}) != len(objects):
        raise ValueError("Object IDs must be unique")
    encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))  # noqa: E731
    tokens = encode("RL-MAC-MKG\nNULL target\n")
    positions = [len(tokens) - 1]
    identifiers: list[str | None] = [None]
    tokens.extend(encode("State: " + summary + "\n")[: min(256, max_tokens // 4)])
    footer = encode("\nSelect operation and object.\n")
    feasible = set(feasible_actions)
    names = tuple(operation_names)
    if ("stop", None) not in feasible or any(operation not in names for operation, _ in feasible):
        raise ValueError("Feasible actions must use known operations and include stop")
    for identifier, description in objects:
        if not identifier or not any(target == identifier for _, target in feasible):
            continue
        prefix = encode(f"\nObject {identifier}: ")
        content = encode(description)[:192]
        available = max_tokens - len(tokens) - len(footer)
        if available < len(prefix) + 1:
            continue
        tokens.extend(prefix + content[: available - len(prefix)])
        positions.append(len(tokens) - 1)
        identifiers.append(identifier)
    tokens.extend(footer)
    mask = tuple(
        tuple((operation, identifier) in feasible for identifier in identifiers)
        for operation in names
    )
    return DecisionContext(
        tuple(tokens),
        tuple(identifiers),
        tuple(positions),
        mask,
        names,
        state_hash,
        evidence_hash,
        specialist_signature,
    )
