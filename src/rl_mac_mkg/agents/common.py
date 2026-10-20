"""Shared frozen versions, measured invocation metadata, and provenance helpers."""

import time
import uuid
from dataclasses import replace

import requests

from ..records import SourceAnchor, SpecialistResult, stable_hash


def version(role, backend=None, **settings):
    return (
        role
        + ":"
        + stable_hash(
            {"role": role, "backend": getattr(backend, "version", None), "contract": 1, **settings}
        )
    )


def anchor_for(source, **fields):
    return SourceAnchor(
        source.source_id,
        source.url,
        source.retrieved_at,
        source.content_hash,
        document_id=source.document_id,
        image_id=source.image_id,
        **fields,
    )


class Specialist:
    """Attach request metadata once; every visible invocation is attempted at most once."""

    def __call__(self, action, state):
        started, request_id = time.perf_counter(), uuid.uuid4().hex
        try:
            result = self.execute(action, state)
        except (TimeoutError, requests.Timeout) as exc:
            result = SpecialistResult("timeout", message=str(exc) or "Specialist timed out")
        except requests.RequestException as exc:
            result = SpecialistResult("error", message=f"{type(exc).__name__}: {exc}")
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result = SpecialistResult("invalid", message=str(exc))
        except Exception as exc:
            result = SpecialistResult("error", message=f"{type(exc).__name__}: {exc}")
        usage = dict(result.usage)
        usage["latency_seconds"] = max(
            time.perf_counter() - started, usage.get("latency_seconds", 0)
        )
        return replace(result, request_id=request_id, specialist_version=self.version, usage=usage)


def response_usage(response):
    usage = {"latency_seconds": response.latency_seconds}
    for field in ("prompt_tokens", "completion_tokens"):
        if getattr(response, field) is not None:
            usage[field] = getattr(response, field)
    return usage


def acquired_anchors(state):
    return tuple(
        a for evidence in state.evidence if evidence.status == "success" for a in evidence.anchors
    )


def anchor_is_subspan(anchor, allowed):
    for old in allowed:
        if (anchor.source_id, anchor.content_hash, anchor.image_id, anchor.document_id) != (
            old.source_id,
            old.content_hash,
            old.image_id,
            old.document_id,
        ):
            continue
        if anchor.document_id is not None and old.start <= anchor.start < anchor.end <= old.end:
            return True
        if anchor.image_id is not None:
            if old.region is None:
                return True
            if anchor.region is not None:
                x0, y0, x1, y1 = anchor.region
                a0, b0, a1, b1 = old.region
                if a0 <= x0 < x1 <= a1 and b0 <= y0 < y1 <= b1:
                    return True
    return False
