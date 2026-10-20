"""Source loading, raw-text chunking, duplicate grouping, and leakage-safe splits."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..records import Record, SourceRecord

PARTITIONS = ("specialist", "verifier", "ppo", "calibration", "validation", "test")


def load_manifest(path: str | Path) -> list[SourceRecord]:
    """Load source JSONL and check original file bytes before decoding UTF-8."""
    manifest_path = Path(path).resolve()
    records, seen = [], set()
    with manifest_path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Source row must be an object")
                if row.get("path") is not None:
                    source_path = (manifest_path.parent / row["path"]).resolve()
                    raw = source_path.read_bytes()
                    if hashlib.sha256(raw).hexdigest() != row.get("content_hash"):
                        raise ValueError("Source content hash mismatch")
                    row["path"] = str(source_path)
                    if row.get("kind") == "text":
                        original = raw.decode("utf-8")
                        if "text" in row and row["text"] != original:
                            raise ValueError("Inline text differs from source bytes")
                        row["text"] = original
                    elif row.get("kind") == "image":
                        from PIL import Image

                        with Image.open(source_path) as image:
                            row.setdefault("width", image.width)
                            row.setdefault("height", image.height)
                            image.verify()
                record = SourceRecord.from_dict(row)
                if record.source_id in seen:
                    raise ValueError("Duplicate source_id")
                seen.add(record.source_id)
                records.append(record)
            except (ValueError, TypeError, OSError, UnicodeError) as exc:
                raise ValueError(f"Invalid source manifest line {number}: {exc}") from exc
    return records


@dataclass(frozen=True)
class TextChunk(Record):
    start: int
    end: int
    text: str


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 100) -> list[TextChunk]:
    """Split on nearby raw boundaries without changing character positions."""
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    if type(max_chars) is not int or type(overlap) is not int or not 0 <= overlap < max_chars:
        raise ValueError("Require 0 <= overlap < max_chars")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            floor = start + max(overlap + 1, max_chars // 2)
            for offset in range(end, floor - 1, -1):
                if text[offset - 1].isspace() or text[offset - 1] in ".!?;。！？；":
                    end = offset
                    break
        chunks.append(TextChunk(start, end, text[start:end]))
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


class _UnionFind:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def duplicate_groups(
    texts: dict[str, str],
    threshold: float = 0.85,
    embeddings: dict[str, Any] | None = None,
    embedding_threshold: float = 0.98,
) -> dict[str, str]:
    """Group exact and near duplicates transitively using token shingles and optional vectors.

    Text normalization is confined to comparison; source text and anchors are untouched.
    Exhaustive Jaccard checks make the grouping independent of approximate search order.
    """
    if not 0 <= threshold <= 1 or not 0 <= embedding_threshold <= 1:
        raise ValueError("Duplicate thresholds must lie in [0, 1]")
    keys = sorted(texts)
    uf = _UnionFind(keys)
    normalized = {key: re.findall(r"\w+", texts[key].casefold()) for key in keys}
    shingles = {}
    for key, tokens in normalized.items():
        n = min(3, len(tokens))
        shingles[key] = {tuple(tokens[i : i + n]) for i in range(max(1, len(tokens) - n + 1))}
    for index, a in enumerate(keys):
        for b in keys[index + 1 :]:
            sa, sb = shingles[a], shingles[b]
            similarity = len(sa & sb) / max(1, len(sa | sb))
            duplicate = normalized[a] == normalized[b] or similarity >= threshold
            if embeddings is not None and a in embeddings and b in embeddings:
                va, vb = embeddings[a], embeddings[b]
                if len(va) != len(vb) or len(va) == 0:
                    raise ValueError("Duplicate embeddings must have equal positive dimensions")
                if not all(math.isfinite(float(v)) for v in list(va) + list(vb)):
                    raise ValueError("Duplicate embeddings must be finite")
                norm = math.sqrt(sum(float(v) ** 2 for v in va) * sum(float(v) ** 2 for v in vb))
                cosine = (
                    sum(float(x) * float(y) for x, y in zip(va, vb, strict=True)) / norm
                    if norm
                    else 0
                )
                duplicate |= cosine >= embedding_threshold
            if duplicate:
                uf.join(a, b)
    return {key: "dup:" + uf.find(key) for key in keys}


def minhash_signature(text: str, num_perm: int = 128, seed: int = 42) -> tuple[int, ...]:
    """Return a stable seeded MinHash signature for normalized word shingles."""
    if type(num_perm) is not int or num_perm < 1:
        raise ValueError("num_perm must be a positive integer")
    words = re.findall(r"\w+", text.casefold())
    size = min(3, len(words))
    shingles = {" ".join(words[i : i + size]) for i in range(max(1, len(words) - size + 1))}
    prime = (1 << 61) - 1
    values = [
        int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % prime
        for value in shingles
    ]
    rng = random.Random(seed)
    permutations = [(rng.randrange(1, prime), rng.randrange(prime)) for _ in range(num_perm)]
    return tuple(min((a * value + b) % prime for value in values) for a, b in permutations)


def perceptual_hash(path: str | Path, hash_size: int = 8) -> str:
    """Compute a horizontal difference hash after image integrity validation."""
    from PIL import Image, ImageOps

    if type(hash_size) is not int or hash_size < 1:
        raise ValueError("hash_size must be positive")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image = (
            ImageOps.exif_transpose(image)
            .convert("L")
            .resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        )
        values = image.tobytes()
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            offset = row * (hash_size + 1) + col
            bits = (bits << 1) | int(values[offset] > values[offset + 1])
    return format(bits, f"0{math.ceil(hash_size * hash_size / 4)}x")


def image_duplicate_groups(
    images: dict[str, str | Path],
    max_hamming: int = 4,
    embeddings: dict[str, Any] | None = None,
    embedding_threshold: float = 0.98,
) -> dict[str, str]:
    """Group image identities with dHash distance and optional CLIP cosine similarity."""
    if type(max_hamming) is not int or not 0 <= max_hamming <= 64:
        raise ValueError("max_hamming must lie in [0, 64]")
    if not 0 <= embedding_threshold <= 1:
        raise ValueError("embedding_threshold must lie in [0, 1]")
    keys, hashes = (
        sorted(images),
        {key: int(perceptual_hash(path), 16) for key, path in images.items()},
    )
    uf = _UnionFind(keys)
    for index, a in enumerate(keys):
        for b in keys[index + 1 :]:
            same = (hashes[a] ^ hashes[b]).bit_count() <= max_hamming
            if embeddings is not None and a in embeddings and b in embeddings:
                va, vb = embeddings[a], embeddings[b]
                if len(va) != len(vb) or len(va) == 0:
                    raise ValueError("Image embeddings must have equal positive dimensions")
                values = list(va) + list(vb)
                if not all(math.isfinite(float(value)) for value in values):
                    raise ValueError("Image embeddings must be finite")
                norm = math.sqrt(sum(float(v) ** 2 for v in va) * sum(float(v) ** 2 for v in vb))
                cosine = (
                    sum(float(x) * float(y) for x, y in zip(va, vb, strict=True)) / norm
                    if norm
                    else 0
                )
                same |= cosine >= embedding_threshold
            if same:
                uf.join(a, b)
    return {key: "imgdup:" + uf.find(key) for key in keys}


def _counts(total: int, fractions: tuple[float, ...]) -> list[int]:
    raw = [total * f for f in fractions]
    counts = [math.floor(v) for v in raw]
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return counts


def split_entities(entity_ids, seed: int = 42) -> dict[str, list[str]]:
    entities = sorted(entity_ids)
    if any(not isinstance(value, str) or not value for value in entities):
        raise ValueError("Entity IDs must be nonempty strings")
    if len(entities) != len(set(entities)):
        raise ValueError("Cannot split duplicate entity IDs")
    random.Random(seed).shuffle(entities)
    ntrain, ncal, nval, ntest = _counts(len(entities), (0.6, 0.1, 0.1, 0.2))
    nsp, nver, nppo = _counts(ntrain, (0.35, 0.25, 0.4))
    sizes = (nsp, nver, nppo, ncal, nval, ntest)
    result, start = {}, 0
    for partition, size in zip(PARTITIONS, sizes, strict=True):
        result[partition] = sorted(entities[start : start + size])
        start += size
    return result


def _entity_partitions(splits) -> dict[str, str]:
    mapping = {}
    for partition, ids in splits.items():
        if partition not in PARTITIONS:
            raise ValueError(f"Unknown partition: {partition}")
        for entity_id in ids:
            if entity_id in mapping:
                raise ValueError(f"Entity leakage: {entity_id}")
            mapping[entity_id] = partition
    return mapping


def _source_keys(source: SourceRecord):
    yield "source", source.source_id
    yield "page", source.url
    yield "hash", source.content_hash
    if source.document_id:
        yield "document", source.document_id
    if source.image_id:
        yield "image", source.image_id
    if source.duplicate_group:
        yield "duplicate", source.duplicate_group


def assign_sources(sources: list[SourceRecord], splits) -> list[SourceRecord]:
    """Give shared pages/images/duplicate components to their earliest partition.

    Entity associations from other partitions are removed, explicitly masking those uses.
    A component member can consequently have an empty entity_ids tuple.
    """
    entity_partition = _entity_partitions(splits)
    uf, owners = _UnionFind(range(len(sources))), {}
    for index, source in enumerate(sources):
        for key in _source_keys(source):
            if key in owners:
                uf.join(index, owners[key])
            owners[key] = index
    groups = defaultdict(list)
    for index, source in enumerate(sources):
        groups[uf.find(index)].append(source)
    assignment = {}
    for group, records in groups.items():
        partitions = {
            entity_partition[e] for s in records for e in s.entity_ids if e in entity_partition
        }
        if not partitions:
            raise ValueError("Source component has no entity assigned to a partition")
        assignment[group] = min(partitions, key=PARTITIONS.index)
    result = []
    for index, source in enumerate(sources):
        partition = assignment[uf.find(index)]
        ids = tuple(e for e in source.entity_ids if entity_partition.get(e) == partition)
        result.append(replace(source, partition=partition, entity_ids=ids))
    return result


def audit_leakage(sources: list[SourceRecord], splits) -> list[str]:
    findings = []
    try:
        entity_partition = _entity_partitions(splits)
    except ValueError as exc:
        findings.append(str(exc))
        entity_partition = {}
    seen = defaultdict(set)
    for source in sources:
        if source.partition not in PARTITIONS:
            findings.append(f"Unassigned source: {source.source_id}")
        for entity_id in source.entity_ids:
            if entity_partition.get(entity_id) != source.partition:
                findings.append(f"Entity/source partition mismatch: {entity_id}/{source.source_id}")
        for key in _source_keys(source):
            seen[key].add(source.partition)
    for (kind, identifier), partitions in sorted(seen.items()):
        if len(partitions) > 1:
            findings.append(f"{kind} leakage: {identifier} spans {sorted(partitions)}")
    return sorted(set(findings))


__all__ = [
    "PARTITIONS",
    "TextChunk",
    "assign_sources",
    "audit_leakage",
    "chunk_text",
    "duplicate_groups",
    "image_duplicate_groups",
    "load_manifest",
    "minhash_signature",
    "perceptual_hash",
    "split_entities",
]
