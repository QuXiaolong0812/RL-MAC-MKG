import hashlib

import pytest

from rl_mac_mkg.environment import Environment, RewardConfig
from rl_mac_mkg.graph import PredicateRule, Schema
from rl_mac_mkg.records import (
    Action,
    Candidate,
    Entity,
    Operation,
    SourceAnchor,
    SourceRecord,
    SpecialistResult,
    Verification,
)
from rl_mac_mkg.workflows import (
    FixedRoutePolicy,
    collect_episode,
    fit_action_costs,
    observation_for,
)


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1] * len(text.split())


def make_environment():
    text = "Evidence available only after retrieval."
    source = SourceRecord(
        "s",
        ("a",),
        "https://example.org/s",
        "2026-10-19T00:00:00Z",
        "CC0",
        hashlib.sha256(text.encode()).hexdigest(),
        "text",
        text=text,
        document_id="d",
    )
    anchor = SourceAnchor(
        "s",
        source.url,
        source.retrieved_at,
        source.content_hash,
        document_id="d",
        start=0,
        end=len(text),
        quote=text,
    )
    candidate = Candidate("c", "a", "isA", "b", (anchor,))

    def dispatch(action, state):
        if action.operation is Operation.RETRIEVE_TEXT:
            return SpecialistResult("success", anchors=(anchor,), specialist_version="v")
        if action.operation is Operation.RETRIEVE_IMAGE:
            return SpecialistResult("empty", specialist_version="v")
        if action.operation is Operation.GENERATE_TEXT:
            return SpecialistResult("success", candidates=(candidate,), specialist_version="v")
        if action.operation is Operation.VERIFY:
            return SpecialistResult(
                "success", verification=Verification("c", 1, 0.99), specialist_version="v"
            )
        return SpecialistResult("empty", specialist_version="v")

    costs = {op: 0 if op is Operation.STOP else 1 for op in Operation}
    return Environment(
        [Entity("a", "Alpha"), Entity("b", "Beta")],
        [source],
        Schema({"isA": PredicateRule(("entity",), ("entity",))}),
        RewardConfig(1, 1, 1, costs),
        dispatch,
        budget=12,
        seed_entity_ids=["a"],
        specialist_versions={"verify": "v"},
    )


def test_unretrieved_source_text_is_not_in_controller_context():
    environment = make_environment()
    seen = []

    class RecordingTokenizer:
        def encode(self, text, add_special_tokens=False):
            seen.append(text)
            return [1] * len(text.split())

    observation_for(environment, RecordingTokenizer())
    assert "Evidence available only after retrieval" not in " ".join(seen)
    environment.step(Action(Operation.RETRIEVE_TEXT, "a"))
    seen.clear()
    observation_for(environment, RecordingTokenizer())
    assert "Evidence available only after retrieval" in " ".join(seen)


def test_fixed_route_exports_provenance_graph_and_ledger():
    environment = make_environment()
    steps, trace = collect_episode(environment, FixedRoutePolicy(), WordTokenizer())
    assert environment.state.done
    assert ("a", "isA", "b") in environment.state.graph.triples()
    assert trace
    assert environment.state.graph.assertions
    assert not steps


def test_cost_estimation_requires_validation_and_complete_operation_coverage():
    rows = [
        {
            "partition": "validation",
            "operation": op.value,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "latency_seconds": 2,
            "retrieval_calls": 1,
        }
        for op in Operation
        if op is not Operation.STOP
    ]
    costs = fit_action_costs(rows, token_weight=1, latency_weight=1, retrieval_weight=1)
    assert costs["stop"] == 0
    assert all(0.02 <= value <= 1 for key, value in costs.items() if key != "stop")
    rows[0]["partition"] = "test"
    with pytest.raises(ValueError, match="validation"):
        fit_action_costs(rows, token_weight=1, latency_weight=1, retrieval_weight=1)
