import numpy as np
import pytest

from rl_mac_mkg.evaluation.calibration import TemperatureScaler, calibration_metrics
from rl_mac_mkg.evaluation.metrics import construction_metrics, retrieval_metrics, selective_metrics
from rl_mac_mkg.evaluation.statistics import bootstrap_interval, holm_correction, paired_comparison


def test_binary_calibration_metrics_include_tie_aware_auroc():
    result = calibration_metrics([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    assert result["auroc"] == pytest.approx(0.5)
    assert result["brier"] == pytest.approx(0.25)
    assert result["nll"] == pytest.approx(np.log(2))
    assert result["ece"] == 0
    assert calibration_metrics([1, 1], [0.8, 0.9])["auroc"] is None
    with pytest.raises(ValueError):
        calibration_metrics([1], [1.5])


def test_temperature_fit_is_calibration_only_and_improves_nll():
    logits = [8, 8, -8, -8]
    labels = [1, 0, 0, 1]
    scaler = TemperatureScaler.fit(logits, labels, partition="calibration")
    assert scaler.temperature > 1
    before = calibration_metrics(labels, TemperatureScaler().predict(logits))["nll"]
    after = calibration_metrics(labels, scaler.predict(logits))["nll"]
    assert after < before
    with pytest.raises(ValueError, match="calibration"):
        TemperatureScaler.fit(logits, labels, partition="test")


def test_construction_scores_distinguish_relations_and_images():
    predictions = [
        ("a", "isA", "b", "relation"),
        ("a", "isA", "c", "relation"),
        ("a", "hasImage", "i", "image"),
    ]
    references = [("a", "isA", "b", "relation"), ("a", "hasImage", "i", "image")]
    result = construction_metrics(predictions, references)
    assert result["relations"]["precision"] == 0.5
    assert result["relations"]["recall"] == 1
    assert result["relations"]["f1"] == pytest.approx(2 / 3)
    assert result["images"]["f1"] == 1
    assert construction_metrics([], [])["relations"]["f1"] is None


def test_selective_risk_and_retrieval_do_not_fill_missing_labels():
    result = selective_metrics({"a", "b", "unannotated"}, {"a": True, "b": False, "c": True})
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["risk"] == 0.5
    assert result["unannotated_acceptances"] == 1
    result = retrieval_metrics(
        {"q1": ["x", "a"], "q2": ["b", "x"]}, {"q1": {"a"}, "q2": {"b"}}, k=1
    )
    assert result["recall_at_k"] == 0.5
    assert result["mrr"] == 0.75


def test_statistics_are_paired_reproducible_and_holm_monotone():
    first = bootstrap_interval([1, 2, 3, 4], resamples=200, seed=8)
    assert first == bootstrap_interval([1, 2, 3, 4], resamples=200, seed=8)
    assert first[0] <= 2.5 <= first[1]
    identical = paired_comparison([1, 2, 3], [1, 2, 3], resamples=200, seed=0)
    assert identical["p_value"] == 1
    assert identical["paired_cliffs_delta"] == 0
    assert holm_correction([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
