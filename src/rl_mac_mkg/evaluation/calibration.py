"""Binary temperature scaling fitted exclusively on calibration labels."""

import math
from dataclasses import dataclass

import numpy as np


def _binary_inputs(labels, probabilities):
    y, p = np.asarray(labels, dtype=float), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError("Labels and probabilities must be nonempty equal-length vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected binary labels and probabilities in [0, 1]")
    return y, p


def calibration_metrics(labels, probabilities, bins=15):
    y, p = _binary_inputs(labels, probabilities)
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    assignments = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            ece += float(selected.mean() * abs(p[selected].mean() - y[selected].mean()))
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    positives = int(y.sum())
    negatives = len(y) - positives
    auroc = None
    if positives and negatives:
        order = np.argsort(p, kind="stable")
        ranks = np.empty(len(y), dtype=float)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and p[order[end]] == p[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
        auroc = float(
            (ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
        )
    return {
        "ece": ece,
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        "auroc": auroc,
        "count": len(y),
    }


@dataclass(frozen=True)
class TemperatureScaler:
    temperature: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be finite and positive")

    def predict(self, logits):
        logits = np.asarray(logits, dtype=float)
        if not np.isfinite(logits).all():
            raise ValueError("Logits must be finite")
        scaled = logits / self.temperature
        return np.exp(-np.logaddexp(0, -scaled))

    @classmethod
    def fit(cls, logits, labels, *, partition, iterations=120):
        if partition != "calibration":
            raise ValueError("Temperature fitting requires the calibration partition")
        x, y = np.asarray(logits, dtype=float), np.asarray(labels, dtype=float)
        if x.ndim != 1 or x.shape != y.shape or not len(x):
            raise ValueError("Expected nonempty aligned logits and labels")
        if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Expected finite logits and binary labels")

        def objective(log_temperature):
            scaled = x / math.exp(log_temperature)
            return float(np.mean(np.logaddexp(0, scaled) - y * scaled))

        lower, upper = -5.0, 5.0
        ratio = (math.sqrt(5) - 1) / 2
        left, right = upper - ratio * (upper - lower), lower + ratio * (upper - lower)
        for _ in range(iterations):
            if objective(left) <= objective(right):
                upper, right = right, left
                left = upper - ratio * (upper - lower)
            else:
                lower, left = left, right
                right = lower + ratio * (upper - lower)
        candidates = [-5.0, 0.0, 5.0, (lower + upper) / 2]
        return cls(math.exp(min(candidates, key=objective)))


def binary_uncertainty(probability: float) -> float:
    if not 0 <= probability <= 1:
        raise ValueError("Expected a probability")
    if probability in (0, 1):
        return 0.0
    return -(
        probability * math.log(probability) + (1 - probability) * math.log1p(-probability)
    ) / math.log(2)
