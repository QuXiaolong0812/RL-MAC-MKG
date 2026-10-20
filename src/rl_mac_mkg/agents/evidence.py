"""Incremental retrieval from an immutable local provenance catalog."""

import hashlib
import math
from pathlib import Path

import numpy as np

from ..data import chunk_text
from ..records import Operation, SpecialistResult
from .common import Specialist, anchor_for, version


def perceptual_image_hash(path):
    """64-bit pHash from an orthonormal two-dimensional DCT of a 32-pixel image."""
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        pixels = np.asarray(
            ImageOps.exif_transpose(image).convert("L").resize((32, 32)), dtype=float
        )
    coords = np.arange(32)
    basis = np.cos(np.pi * (2 * coords[None, :] + 1) * coords[:, None] / 64)
    basis[0] /= math.sqrt(2)
    low = (basis @ pixels @ basis.T)[:8, :8].reshape(-1)
    threshold = np.median(low[1:])
    bits = low > threshold
    # The DC coefficient contains mean brightness and is excluded from comparisons.
    bits[0] = False
    return sum(int(bit) << index for index, bit in enumerate(bits))


class EvidenceAgent(Specialist):
    def __init__(
        self,
        schema=None,
        *,
        clip_backend=None,
        allowed_licenses=("CC-BY-4.0", "CC0-1.0", "public-domain"),
        min_image_size=(64, 64),
        max_image_pixels=40_000_000,
        chunk_chars=2000,
        chunk_overlap=100,
        max_hamming=4,
    ):
        chunk_text("", chunk_chars, chunk_overlap)
        if len(min_image_size) != 2 or any(type(v) is not int or v < 1 for v in min_image_size):
            raise ValueError("min_image_size requires two positive integer dimensions")
        if type(max_image_pixels) is not int or max_image_pixels < 1:
            raise ValueError("max_image_pixels must be a positive integer")
        if type(max_hamming) is not int or not 0 <= max_hamming <= 63:
            raise ValueError("max_hamming must lie in [0, 63]")
        self.clip_backend, self.allowed_licenses = clip_backend, frozenset(allowed_licenses)
        self.min_image_size, self.max_image_pixels = tuple(min_image_size), max_image_pixels
        self.chunk_chars, self.chunk_overlap, self.max_hamming = (
            chunk_chars,
            chunk_overlap,
            max_hamming,
        )
        self.version = version(
            "PEA",
            clip_backend,
            schema=schema,
            licenses=sorted(self.allowed_licenses),
            min_image_size=min_image_size,
            max_image_pixels=max_image_pixels,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
            max_hamming=max_hamming,
            image_hash="dct-phash-63-v1",
        )

    def _image_issue(self, source):
        from PIL import Image

        if (
            not source.eligible
            or source.license not in self.allowed_licenses
            or source.safety_status != "safe"
        ):
            return (
                "unsafe",
                "Image requires an allowed license, eligible provenance, and safe status",
            )
        try:
            if not source.path:
                raise ValueError("Image requires a local original file")
            if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != source.content_hash:
                raise ValueError("Image source content hash mismatch")
            with Image.open(source.path) as image:
                if image.width * image.height > self.max_image_pixels:
                    raise ValueError("Image exceeds max_image_pixels before pixel decoding")
                if image.width < self.min_image_size[0] or image.height < self.min_image_size[1]:
                    raise ValueError("Image is below minimum resolution")
                if (source.width is not None and image.width != source.width) or (
                    source.height is not None and image.height != source.height
                ):
                    raise ValueError("Image dimensions differ from catalog metadata")
                image.verify()
            return None
        except (OSError, ValueError, SyntaxError) as exc:
            return "invalid", str(exc)

    def execute(self, action, state):
        if action.operation not in (Operation.RETRIEVE_TEXT, Operation.RETRIEVE_IMAGE):
            raise ValueError("PEA accepts retrieval operations only")
        if action.target_id not in state.entities:
            raise ValueError("Retrieval target entity is unknown")
        kind = "text" if action.operation is Operation.RETRIEVE_TEXT else "image"
        cursor = state.retrieval_cursors.get(f"{action.target_id}:{kind}", 0)
        sources = sorted(
            (
                s
                for s in state.sources.values()
                if action.target_id in s.entity_ids and s.kind == kind
            ),
            key=lambda s: s.source_id,
        )
        anchors, excluded = [], []
        if kind == "text":
            seen = set()
            for source in sources:
                if not source.eligible:
                    excluded.append(("invalid", f"Ineligible text source: {source.source_id}"))
                    continue
                group = source.duplicate_group or source.content_hash
                if group in seen:
                    continue
                seen.add(group)
                anchors.extend(
                    anchor_for(source, start=c.start, end=c.end, quote=c.text)
                    for c in chunk_text(source.text, self.chunk_chars, self.chunk_overlap)
                )
        else:
            eligible = []
            for source in sources:
                issue = self._image_issue(source)
                if issue:
                    excluded.append((issue[0], source.source_id + ": " + issue[1]))
                else:
                    eligible.append(source)
            scores = [0.0] * len(eligible)
            if eligible and self.clip_backend is not None:
                scores = list(
                    self.clip_backend.score(
                        state.entities[action.target_id].name, [s.path for s in eligible]
                    )
                )
                if len(scores) != len(eligible) or any(not math.isfinite(float(v)) for v in scores):
                    raise ValueError(
                        "CLIP scores must be finite and match the candidate image count"
                    )
            ranked = sorted(
                zip(eligible, scores, strict=True), key=lambda item: (-item[1], item[0].source_id)
            )
            hashes, groups = [], set()
            for source, _ in ranked:
                phash = perceptual_image_hash(source.path)
                if (source.duplicate_group and source.duplicate_group in groups) or any(
                    (phash ^ old).bit_count() <= self.max_hamming for old in hashes
                ):
                    continue
                hashes.append(phash)
                if source.duplicate_group:
                    groups.add(source.duplicate_group)
                anchors.append(anchor_for(source))
        message = "; ".join(issue[1] for issue in excluded)
        if cursor < len(anchors):
            return SpecialistResult(
                "success",
                anchors=(anchors[cursor],),
                message=message,
                usage={"retrieval_items": float(len(sources))},
            )
        if not anchors and excluded:
            status = "unsafe" if all(item[0] == "unsafe" for item in excluded) else "invalid"
            return SpecialistResult(status, message=message)
        return SpecialistResult(
            "empty", message="No further eligible evidence" + ("; " + message if message else "")
        )
