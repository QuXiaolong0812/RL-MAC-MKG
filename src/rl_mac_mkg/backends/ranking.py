"""Local frozen ranking and type-aware entity canonicalization."""

from pathlib import Path

import numpy as np

from ..records import stable_hash
from .local import asset_fingerprint


def _normalized(vectors, count=None):
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or (count is not None and len(vectors) != count) or vectors.shape[1] == 0:
        raise ValueError("Embedding vectors must be a rectangular matrix matching input count")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding vectors must be finite")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding vectors must have positive norm")
    return vectors / norms


class CLIPBackend:
    """Frozen CLIP cosine ranking; both model and processor load from local files only."""

    def __init__(self, model_path, *, device="cpu", batch_size=16):
        self.model_path, self.device, self.batch_size = Path(model_path), device, batch_size
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("CLIP batch_size must be a positive integer")
        self.version = "CLIP:" + stable_hash(
            {
                "assets": asset_fingerprint(self.model_path),
                "device": device,
                "batch_size": batch_size,
                "ranking": "normalized-cosine-v1",
            }
        )
        self.model = self.processor = None

    def _load(self):
        if self.model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False
        )
        self.processor = CLIPProcessor.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False
        )
        self.model.to(self.device).eval().requires_grad_(False)

    def score(self, query, paths):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("CLIP query must be a nonempty string")
        paths = list(paths)
        if not paths:
            return []
        self._load()
        import torch
        from PIL import Image, ImageOps

        scores = []
        for start in range(0, len(paths), self.batch_size):
            images = []
            try:
                for path in paths[start : start + self.batch_size]:
                    with Image.open(path) as original:
                        images.append(ImageOps.exif_transpose(original).convert("RGB"))
                inputs = self.processor(
                    text=[query], images=images, padding=True, truncation=True, return_tensors="pt"
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    outputs = self.model(**inputs)
                text = _normalized(outputs.text_embeds.detach().float().cpu().numpy(), 1)
                image = _normalized(
                    outputs.image_embeds.detach().float().cpu().numpy(), len(images)
                )
                scores.extend((image @ text[0]).tolist())
            finally:
                for image in images:
                    image.close()
        return scores


class SentenceEncoder:
    """An optional local sentence-transformers encoder used only for canonicalization."""

    def __init__(self, model_path, *, device="cpu"):
        self.model_path, self.device = Path(model_path), device
        self.version = "sentence:" + stable_hash(
            {"assets": asset_fingerprint(self.model_path), "device": device}
        )
        self.model = None

    def encode(self, texts):
        texts = list(texts)
        if not texts or any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("Sentence encoder requires nonempty text strings")
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                str(self.model_path),
                device=self.device,
                local_files_only=True,
                trust_remote_code=False,
            )
            self.model.eval().requires_grad_(False)
        return _normalized(
            self.model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            ),
            len(texts),
        )


class FrozenCanonicalizer:
    """Exact identities first; semantic resolution needs declared compatible entity types.

    A semantic match must clear both a cosine threshold and a runner-up margin. Unknown
    values passed to __call__ remain untouched, so literal values and predicate names
    cannot be merged merely because they resemble an entity name.
    """

    def __init__(self, entities, *, encoder=None, threshold=0.95, ambiguity_margin=0.05):
        if not 0 <= threshold <= 1 or not 0 <= ambiguity_margin <= 1:
            raise ValueError("Canonicalization threshold and ambiguity margin must lie in [0, 1]")
        entities = list(entities.values() if isinstance(entities, dict) else entities)
        self.entities = {entity.entity_id: entity for entity in entities}
        if len(entities) != len(self.entities):
            raise ValueError("Duplicate canonical entity IDs")
        self.encoder, self.threshold, self.ambiguity_margin = encoder, threshold, ambiguity_margin
        self.aliases = {}
        for entity in entities:
            for name in (entity.name, *entity.aliases):
                self.aliases.setdefault(name.casefold(), set()).add(entity.entity_id)
        self.order = sorted(self.entities)
        self.embeddings = None
        if encoder is not None and self.order:
            self.embeddings = _normalized(
                encoder.encode([self.entities[k].name for k in self.order]), len(self.order)
            )
        self.vocabulary_hash = stable_hash([self.entities[k] for k in self.order])
        self.version = "canonicalizer:" + stable_hash(
            {
                "encoder": getattr(encoder, "version", None),
                "threshold": threshold,
                "ambiguity_margin": ambiguity_margin,
                "matching": "exact-then-type-filtered-margin-v1",
            }
        )

    def __call__(self, value):
        if value in self.entities:
            return value
        matches = self.aliases.get(value.casefold(), set())
        if len(matches) > 1:
            raise ValueError("Ambiguous exact entity alias")
        return next(iter(matches)) if matches else value

    def resolve(self, value, *, types):
        if not types:
            raise ValueError("Semantic canonicalization requires allowed entity types")
        if value in self.entities:
            if self.entities[value].entity_type not in types:
                raise ValueError("Canonical entity type is incompatible")
            return value
        exact = self.aliases.get(value.casefold(), set())
        matches = [key for key in exact if self.entities[key].entity_type in types]
        if len(matches) > 1:
            raise ValueError("Ambiguous exact entity alias")
        if matches:
            return matches[0]
        if exact:
            raise ValueError("Exact entity type is incompatible")
        if self.encoder is None or self.embeddings is None:
            raise ValueError("Unknown entity: no semantic encoder is configured")
        query = _normalized(self.encoder.encode([value]), 1)
        if query.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Canonicalization embeddings must have equal dimensions")
        scores = self.embeddings @ query[0]
        eligible = sorted(
            (
                (float(scores[index]), key)
                for index, key in enumerate(self.order)
                if self.entities[key].entity_type in types
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if not eligible or eligible[0][0] < self.threshold:
            raise ValueError("Unknown entity: semantic match does not reach threshold")
        if len(eligible) > 1 and eligible[0][0] - eligible[1][0] < self.ambiguity_margin:
            raise ValueError("Ambiguous semantic entity match")
        return eligible[0][1]
