from dataclasses import replace

import pytest
from test_workflows import make_environment

from rl_mac_mkg.evaluation.perturbations import perturb_sources


def test_missing_text_is_seeded_and_original_records_immutable():
    source = next(iter(make_environment().state.sources.values()))
    sources = [
        replace(source, source_id=f"s{i}", url=f"https://example.org/{i}") for i in range(10)
    ]
    first, audit = perturb_sources(sources, "missing_text", rate=0.5, seed=8)
    second, _ = perturb_sources(sources, "missing_text", rate=0.5, seed=8)
    assert first == second
    assert len(first) == 5
    assert len(sources) == 10
    assert len(audit["removed_source_ids"]) == 5


def test_text_noise_requires_explicit_adversarial_edits():
    source = next(iter(make_environment().state.sources.values()))
    with pytest.raises(ValueError, match="replacement"):
        perturb_sources([source], "noisy_text", rate=1, seed=0)
