"""Metrics use supplied annotations and return None for undefined denominators."""

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np


def precision_recall_f1(predicted, reference):
    predicted, reference = set(predicted), set(reference)
    true_positive = len(predicted & reference)
    precision = true_positive / len(predicted) if predicted else None
    recall = true_positive / len(reference) if reference else None
    f1 = 2 * true_positive / (len(predicted) + len(reference)) if predicted or reference else None
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "predicted": len(predicted),
        "reference": len(reference),
    }


def construction_metrics(predictions, references):
    """Match canonical (head, predicate, tail, relation|image) keys."""
    predictions, references = list(predictions), list(references)
    for value in predictions + references:
        if len(value) != 4 or value[3] not in {"relation", "image"}:
            raise ValueError("Expected four-field canonical triple keys")
    return {
        name: precision_recall_f1(
            [tuple(item[:3]) for item in predictions if item[3] == kind],
            [tuple(item[:3]) for item in references if item[3] == kind],
        )
        for name, kind in (("relations", "relation"), ("images", "image"))
    }


def selective_metrics(accepted_ids, annotated_correctness: Mapping[str, bool]):
    if any(type(value) is not bool for value in annotated_correctness.values()):
        raise ValueError("Selective labels must be booleans")
    accepted = set(accepted_ids)
    annotated = accepted & set(annotated_correctness)
    errors = sum(not annotated_correctness[identifier] for identifier in annotated)
    return {
        "coverage": len(annotated) / len(annotated_correctness) if annotated_correctness else None,
        "risk": errors / len(annotated) if annotated else None,
        "annotated_acceptances": len(annotated),
        "errors": errors,
        "unannotated_acceptances": len(accepted - set(annotated_correctness)),
    }


def retrieval_metrics(
    rankings: Mapping[str, Sequence[str]], relevant: Mapping[str, set[str]], k=10
):
    if k < 1:
        raise ValueError("k must be positive")
    recalls, reciprocal_ranks = [], []
    for query, positives in relevant.items():
        if not positives:
            continue
        ranking = list(rankings.get(query, []))
        if len(set(ranking)) != len(ranking):
            raise ValueError("Retrieval rankings must not contain duplicate IDs")
        recalls.append(len(set(ranking[:k]) & set(positives)) / len(positives))
        first = next(
            (index for index, identifier in enumerate(ranking, 1) if identifier in positives), None
        )
        reciprocal_ranks.append(1 / first if first else 0.0)
    return {
        "recall_at_k": float(np.mean(recalls)) if recalls else None,
        "mrr": float(np.mean(reciprocal_ranks)) if recalls else None,
        "k": k,
        "evaluated_queries": len(recalls),
    }


def cosine_rankings(query_ids, queries, item_ids, items):
    queries, items = np.asarray(queries, dtype=float), np.asarray(items, dtype=float)
    if (
        queries.ndim != 2
        or items.ndim != 2
        or queries.shape[1] != items.shape[1]
        or len(query_ids) != len(queries)
        or len(item_ids) != len(items)
    ):
        raise ValueError("Embedding dimensions and IDs must agree")
    if not np.isfinite(queries).all() or not np.isfinite(items).all():
        raise ValueError("Embeddings must be finite")
    q_norm, i_norm = np.linalg.norm(queries, axis=1), np.linalg.norm(items, axis=1)
    if np.any(q_norm == 0) or np.any(i_norm == 0):
        raise ValueError("Cannot rank zero embeddings")
    scores = (queries / q_norm[:, None]) @ (items / i_norm[:, None]).T
    return {
        identifier: [item_ids[index] for index in np.argsort(-scores[row], kind="stable")]
        for row, identifier in enumerate(query_ids)
    }


def graph_structure(
    keys,
    *,
    entities=(),
    provenance=None,
    schema_validity=None,
    contradiction_pairs=(),
    resolution_labels=None,
):
    keys = list(keys)
    nodes = set(entities)
    adjacency = {}
    for head, _, tail, kind in keys:
        nodes.add(head)
        if kind == "relation":
            nodes.add(tail)
            adjacency.setdefault(head, set()).add(tail)
            adjacency.setdefault(tail, set()).add(head)
    unseen, components = set(nodes), []
    while unseen:
        stack = [unseen.pop()]
        component = set(stack)
        while stack:
            for neighbor in adjacency.get(stack.pop(), set()):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(len(component))
    unique = set(tuple(key) for key in keys)
    involved = {tuple(key) for pair in contradiction_pairs for key in pair} & unique
    return {
        "entities": len(nodes),
        "assertions": len(keys),
        "components": len(components),
        "largest_component_ratio": max(components) / len(nodes) if nodes else None,
        "duplicate_rate": (len(keys) - len(unique)) / len(keys) if keys else None,
        "contradiction_rate": len(involved) / len(unique) if unique else None,
        "provenance_coverage": (
            sum(bool(value) for value in provenance) / len(provenance) if provenance else None
        ),
        "schema_consistency": (float(np.mean(schema_validity)) if schema_validity else None),
        "entity_resolution_accuracy": (
            float(np.mean(resolution_labels)) if resolution_labels else None
        ),
    }


def annotation_agreement(ratings):
    """Nominal Krippendorff alpha and Fleiss kappa; missing labels use None."""
    rows = [[label for label in row if label is not None] for row in ratings]
    rows = [row for row in rows if len(row) >= 2]
    if not rows:
        return {"krippendorff_alpha": None, "fleiss_kappa": None}
    total = sum(len(row) for row in rows)
    counts = Counter(label for row in rows for label in row)
    observed_disagreement = (
        sum(
            (len(row) ** 2 - sum(count**2 for count in Counter(row).values())) / (len(row) - 1)
            for row in rows
        )
        / total
    )
    expected_disagreement = (total**2 - sum(count**2 for count in counts.values())) / (
        total * (total - 1)
    )
    alpha = 1 - observed_disagreement / expected_disagreement if expected_disagreement else None
    kappa = None
    if len({len(row) for row in rows}) == 1:
        n = len(rows[0])
        observed = np.mean(
            [(sum(count**2 for count in Counter(row).values()) - n) / (n * (n - 1)) for row in rows]
        )
        expected = sum((count / total) ** 2 for count in counts.values())
        kappa = float((observed - expected) / (1 - expected)) if expected < 1 else None
    return {"krippendorff_alpha": alpha, "fleiss_kappa": kappa}
