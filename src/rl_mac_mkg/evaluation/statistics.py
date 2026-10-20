"""Seeded entity bootstrap and paired randomization statistics."""

import itertools

import numpy as np


def _vector(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Expected a nonempty finite vector")
    return array


def bootstrap_interval(values, *, resamples=10000, confidence=0.95, seed=0):
    values = _vector(values)
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, resamples, 128):
        draws = rng.integers(0, len(values), size=(min(128, resamples - start), len(values)))
        means.extend(values[draws].mean(axis=1).tolist())
    alpha = (1 - confidence) / 2
    return tuple(float(value) for value in np.quantile(means, [alpha, 1 - alpha]))


def paired_comparison(first, second, *, resamples=10000, seed=0):
    first, second = _vector(first), _vector(second)
    if first.shape != second.shape or resamples < 1:
        raise ValueError("Paired observations must align and resamples must be positive")
    differences = first - second
    observed = abs(float(differences.mean()))
    if len(differences) <= 15:
        extremes, total = 0, 0
        for signs in itertools.product([-1, 1], repeat=len(differences)):
            extremes += abs(float(np.mean(differences * signs))) >= observed - 1e-15
            total += 1
        probability = extremes / total
    else:
        rng = np.random.default_rng(seed)
        extremes = 0
        for start in range(0, resamples, 128):
            signs = rng.choice([-1, 1], size=(min(128, resamples - start), len(differences)))
            extremes += int(np.sum(np.abs((signs * differences).mean(axis=1)) >= observed - 1e-15))
        probability = (extremes + 1) / (resamples + 1)
    return {
        "mean_difference": float(differences.mean()),
        "p_value": float(probability),
        "paired_cliffs_delta": float(np.mean(np.sign(differences))),
        "confidence_interval": bootstrap_interval(differences, resamples=resamples, seed=seed),
        "pairs": len(differences),
    }


def holm_correction(p_values):
    values = _vector(p_values)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values)
    result = np.empty_like(values)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        result[index] = previous
    return result.tolist()


def seed_summary(values):
    values = _vector(values)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "seeds": len(values),
    }
