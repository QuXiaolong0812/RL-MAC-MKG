"""Seeded evidence perturbations with explicit derivation records."""

import hashlib
import random
from dataclasses import replace


def perturb_sources(
    sources, condition, *, rate, seed=42, replacements=None, conflicting_sources=None
):
    if not 0 <= rate <= 1:
        raise ValueError("Perturbation rate must lie in [0, 1]")
    if condition not in {
        "missing_text",
        "missing_images",
        "irrelevant_images",
        "noisy_text",
        "contradictory_sources",
    }:
        raise ValueError("Unknown perturbation condition")
    sources = list(sources)
    kind = "image" if condition in {"missing_images", "irrelevant_images"} else "text"
    eligible = sorted(
        (source for source in sources if source.kind == kind), key=lambda source: source.source_id
    )
    selected = set(
        random.Random(seed).sample(
            [source.source_id for source in eligible], round(rate * len(eligible))
        )
    )
    audit = {
        "condition": condition,
        "rate": rate,
        "seed": seed,
        "selected_source_ids": sorted(selected),
        "removed_source_ids": [],
        "derivations": [],
    }
    if condition in {"missing_text", "missing_images"}:
        audit["removed_source_ids"] = sorted(selected)
        return [source for source in sources if source.source_id not in selected], audit
    if condition == "contradictory_sources":
        if not conflicting_sources:
            raise ValueError("Conflicting sources must be supplied with real provenance")
        existing = {source.source_id for source in sources}
        if any(source.source_id in existing for source in conflicting_sources):
            raise ValueError("Conflicting source IDs must be distinct")
        return sources + list(conflicting_sources), audit
    if condition == "noisy_text" and not replacements:
        raise ValueError("Noisy text requires explicit relation-bearing sentence replacements")
    result = []
    for source in sources:
        if source.source_id not in selected:
            result.append(source)
            continue
        if condition == "irrelevant_images":
            alternatives = [
                other
                for other in eligible
                if set(other.entity_ids).isdisjoint(source.entity_ids)
                and other.content_hash != source.content_hash
                and other.partition == source.partition
            ]
            if not alternatives:
                raise ValueError("No unrelated image in the same partition is available")
            other = alternatives[
                random.Random(f"{seed}:{source.source_id}").randrange(len(alternatives))
            ]
            changed = replace(
                other,
                source_id=source.source_id,
                entity_ids=source.entity_ids,
                duplicate_group=other.duplicate_group,
            )
        else:
            edit = replacements.get(source.source_id)
            if not edit:
                raise ValueError(f"Missing text replacement for {source.source_id}")
            start, end, replacement = edit["start"], edit["end"], edit["replacement"]
            if not 0 <= start < end <= len(source.text) or not isinstance(replacement, str):
                raise ValueError("Invalid text replacement span")
            text = source.text[:start] + replacement + source.text[end:]
            changed = replace(
                source,
                text=text,
                path=None,
                content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        audit["derivations"].append(
            {
                "source_id": source.source_id,
                "original_hash": source.content_hash,
                "perturbed_hash": changed.content_hash,
            }
        )
        result.append(changed)
    return result, audit
