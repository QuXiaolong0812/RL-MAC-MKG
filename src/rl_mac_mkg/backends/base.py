"""Backend contracts and strict structured-response parsing."""

import json
import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelResponse:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_seconds: float


class ModelBackend(Protocol):
    version: str

    def generate(self, system: str, user: str, images=()) -> ModelResponse: ...

    def score_binary(self, prompt: str, images=()) -> float: ...


def parse_json_object(content: str) -> dict:
    content = content.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", content, flags=re.DOTALL)
    if fence:
        content = fence.group(1).strip()

    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(content, parse_constant=reject_constant, object_pairs_hook=unique_pairs)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Model response is not a complete JSON object") from error
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value
