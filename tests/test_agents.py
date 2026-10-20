import json
import math
from dataclasses import replace
from hashlib import sha256

import pytest
from PIL import Image

from rl_mac_mkg import agents
from rl_mac_mkg.backends.base import ModelResponse
from rl_mac_mkg.environment import Environment, RewardConfig
from rl_mac_mkg.graph import PredicateRule, Schema
from rl_mac_mkg.records import Action, Candidate, Entity, EvidenceRecord, Operation, SourceRecord


class Backend:
    version = "fixed-test-weights"

    def __init__(self, output=None, logits=()):
        self.output, self.logits, self.calls = output, iter(logits), []

    def generate(self, system, user, images=()):
        self.calls.append((system, user, images))
        if isinstance(self.output, Exception):
            raise self.output
        return ModelResponse(json.dumps(self.output), 11, 7, 0.2)

    def score_binary(self, prompt, images=()):
        self.calls.append((prompt, images))
        return next(self.logits)


def text_source(text="Panda 熊猫\r\nPanda belongs to Ursidae.", source_id="s1"):
    return SourceRecord(
        source_id,
        ("panda",),
        "https://example.org/" + source_id,
        "2026-10-19T10:00:00Z",
        "CC-BY-4.0",
        sha256(text.encode()).hexdigest(),
        "text",
        text=text,
        document_id=source_id,
    )


def image_source(tmp_path, source_id="i1", **overrides):
    path = tmp_path / (source_id + ".png")
    Image.new("RGB", (128, 128), "red").save(path)
    source = SourceRecord(
        source_id,
        ("panda",),
        "https://example.org/" + source_id,
        "2026-10-19T10:00:00Z",
        "CC-BY-4.0",
        sha256(path.read_bytes()).hexdigest(),
        "image",
        path=str(path),
        image_id=source_id,
        width=128,
        height=128,
        safety_status="safe",
    )
    return replace(source, **overrides)


def env(sources=None):
    schema = Schema(
        {
            "family": PredicateRule(("animal",), ("family",), True),
            "color": PredicateRule(("animal",), tail_is_entity=False),
            "hasImage": PredicateRule(("animal",), tail_is_entity=False),
        }
    )
    config = RewardConfig(1, 0.5, 0.2, {op: 0 if op is Operation.STOP else 0.1 for op in Operation})
    return Environment(
        [Entity("panda", "Panda", "animal"), Entity("ursidae", "Ursidae", "family")],
        sources or [text_source()],
        schema,
        config,
    )


def retrieved(environment, operation=Operation.RETRIEVE_TEXT):
    result = agents.EvidenceAgent(min_image_size=(64, 64))(
        Action(operation, "panda"), environment.state
    )
    environment.state.evidence.append(
        EvidenceRecord("e1", Action(operation, "panda"), result.status, 0.1, result.anchors)
    )
    return result


def output(source=None, **changes):
    source = source or text_source()
    row = {
        "head": "Panda",
        "predicate": "family",
        "tail": "Ursidae",
        "support": [
            {
                "source_id": source.source_id,
                "start": 10,
                "end": len(source.text),
                "quote": source.text[10:],
            }
        ],
        "confidence": 0.01,
    }
    row.update(changes)
    return {"status": "success", "candidates": [row]}


def test_pea_chunks_use_original_offsets_and_environment_cursor():
    environment = env([text_source("A 熊猫\r\n" + "other text " * 12)])
    pea = agents.EvidenceAgent(chunk_chars=28, chunk_overlap=3)
    action = Action(Operation.RETRIEVE_TEXT, "panda")
    first = pea(action, environment.state)
    assert first.status == "success" and first.anchors[0].start == 0
    assert first.anchors[0].quote == environment.state.sources["s1"].text[: first.anchors[0].end]
    assert not environment.state.retrieval_cursors
    environment.state.retrieval_cursors["panda:text"] = 1
    second = pea(action, environment.state)
    assert second.anchors[0].start == first.anchors[0].end - 3
    assert first.request_id and first.specialist_version == pea.version
    assert first.usage["latency_seconds"] >= 0


def test_pea_exhaustion_is_explicit_empty():
    environment = env()
    environment.state.retrieval_cursors["panda:text"] = 100
    assert (
        agents.EvidenceAgent()(Action(Operation.RETRIEVE_TEXT, "panda"), environment.state).status
        == "empty"
    )


@pytest.mark.parametrize(
    "change,status",
    [
        ({"license": "unknown"}, "unsafe"),
        ({"safety_status": "unchecked"}, "unsafe"),
        ({"width": 20}, "invalid"),
    ],
)
def test_pea_rejects_ineligible_images_without_touching_originals(tmp_path, change, status):
    source = image_source(tmp_path, **change)
    original = open(source.path, "rb").read()
    environment = env([source])
    result = agents.EvidenceAgent(min_image_size=(64, 64))(
        Action(Operation.RETRIEVE_IMAGE, "panda"), environment.state
    )
    assert result.status == status
    assert open(source.path, "rb").read() == original


def test_pea_integrity_duplicate_filter_and_clip_ranking(tmp_path):
    a, b = image_source(tmp_path, "a"), image_source(tmp_path, "b")

    class Ranker:
        version = "clip-test"

        def score(self, query, paths):
            assert query == "Panda"
            return [1.0 if str(path).endswith("b.png") else 0.0 for path in paths]

    environment = env([a, b])
    pea = agents.EvidenceAgent(clip_backend=Ranker(), min_image_size=(64, 64))
    first = pea(Action(Operation.RETRIEVE_IMAGE, "panda"), environment.state)
    assert first.anchors[0].image_id == "b"
    environment.state.retrieval_cursors["panda:image"] = 1
    assert pea(Action(Operation.RETRIEVE_IMAGE, "panda"), environment.state).status == "empty"
    with open(a.path, "ab") as stream:
        stream.write(b"tampering")
    result = pea(Action(Operation.RETRIEVE_IMAGE, "panda"), env([a]).state)
    assert result.status == "invalid"


def test_tca_maps_names_and_keeps_low_confidence_candidates():
    environment = env()
    retrieved(environment)
    backend = Backend(output())
    result = agents.TextualCandidateAgent(backend, environment.schema)(
        Action(Operation.GENERATE_TEXT, "e1"), environment.state
    )
    assert result.status == "success"
    candidate = result.candidates[0]
    assert (candidate.head, candidate.predicate, candidate.tail) == ("panda", "family", "ursidae")
    assert candidate.generation_confidence == 0.01
    assert candidate.anchors[0].validate(environment.state.sources) == text_source().text[10:]
    assert result.usage["prompt_tokens"] == 11
    assert "original" in backend.calls[0][0].lower()


def test_network_timeouts_and_transport_errors_keep_distinct_observation_statuses():
    import requests

    environment = env()
    retrieved(environment)
    for error, status in (
        (requests.Timeout("timeout"), "timeout"),
        (requests.ConnectionError("offline"), "error"),
    ):
        backend = Backend(error)
        result = agents.TextualCandidateAgent(backend, environment.schema)(
            Action(Operation.GENERATE_TEXT, "e1"), environment.state
        )
        assert result.status == status
        assert len(backend.calls) == 1


@pytest.mark.parametrize("mutation", ["source", "quote", "outside", "extra", "partial"])
def test_tca_rejects_fabricated_support_and_malformed_output_atomically(mutation):
    environment = env()
    retrieved(environment)
    data = output()
    support = data["candidates"][0]["support"][0]
    if mutation == "source":
        support["source_id"] = "invented"
    elif mutation == "quote":
        support["quote"] = "invented support"
    elif mutation == "outside":
        environment.state.evidence[0] = replace(
            environment.state.evidence[0],
            anchors=(
                replace(
                    environment.state.evidence[0].anchors[0],
                    start=10,
                    quote=text_source().text[10:],
                ),
            ),
        )
        support.update(start=0, quote=text_source().text[: support["end"]])
    elif mutation == "extra":
        data["candidates"][0]["unrecognized"] = True
    else:
        data["candidates"].append({"head": "Panda"})
    result = agents.TextualCandidateAgent(Backend(data), environment.schema)(
        Action(Operation.GENERATE_TEXT, "e1"), environment.state
    )
    assert result.status == "parse_error" and not result.candidates


def test_tca_proposes_schema_typed_new_entity():
    environment = env()
    retrieved(environment)
    data = output(tail={"name": "NewFamily", "entity_type": "family"})
    result = agents.TextualCandidateAgent(Backend(data), environment.schema)(
        Action(Operation.GENERATE_TEXT, "e1"), environment.state
    )
    assert result.status == "success"
    assert result.entities[0].entity_type == "family"
    assert result.candidates[0].tail == result.entities[0].entity_id


def test_timeout_records_usage_and_does_not_retry():
    environment = env()
    retrieved(environment)
    backend = Backend(TimeoutError("deadline"))
    agent = agents.TextualCandidateAgent(backend, environment.schema)
    result = agent(Action(Operation.GENERATE_TEXT, "e1"), environment.state)
    assert result.status == "timeout" and len(backend.calls) == 1
    assert result.request_id and result.specialist_version == agent.version
    assert result.usage["latency_seconds"] >= 0


def test_visual_passes_supplied_images_and_rejects_wrong_image_id(tmp_path):
    source = image_source(tmp_path)
    environment = env([source])
    retrieved(environment, Operation.RETRIEVE_IMAGE)
    data = {
        "status": "success",
        "candidates": [
            {
                "head": "Panda",
                "predicate": "hasImage",
                "tail": "i1",
                "support": [{"source_id": "i1", "image_id": "i1", "region": [0, 0, 1, 1]}],
            }
        ],
    }
    backend = Backend(data)
    agent = agents.VisualGroundingAgent(backend, environment.schema)
    result = agent(Action(Operation.GENERATE_VISUAL, "e1"), environment.state)
    assert result.status == "success" and backend.calls[0][2] == (source.path,)
    backend.output["candidates"][0]["support"][0]["image_id"] = "unsupplied"
    assert agent(Action(Operation.GENERATE_VISUAL, "e1"), environment.state).status == "parse_error"


def test_visual_ambiguity_is_preserved(tmp_path):
    environment = env([image_source(tmp_path)])
    retrieved(environment, Operation.RETRIEVE_IMAGE)
    result = agents.VisualGroundingAgent(
        Backend({"status": "ambiguous", "candidates": []}), environment.schema
    )(Action(Operation.GENERATE_VISUAL, "e1"), environment.state)
    assert result.status == "ambiguous" and not result.candidates


def test_revision_refreshes_support_and_preserves_id_incremented_version():
    environment = env()
    evidence = retrieved(environment)
    environment.add_candidate(
        Candidate("c1", "panda", "family", "ursidae", evidence.anchors, version=1)
    )
    result = agents.TextualCandidateAgent(Backend(output()), environment.schema)(
        Action(Operation.REVISE, "c1"), environment.state
    )
    assert result.candidates[0].candidate_id == "c1"
    assert result.candidates[0].version == 2
    assert result.candidates[0].anchors[0].start == 10


def test_verifier_calibrates_binary_logits_without_soft_hard_gate():
    environment = env()
    result = retrieved(environment)
    environment.add_candidate(Candidate("c1", "panda", "family", "ursidae", result.anchors))
    backend = Backend(logits=[-2])
    verifier = agents.CalibratedVerificationAgent(
        backend, environment.schema, factual_temperature=2, visual_temperature=1
    )
    result = verifier(Action(Operation.VERIFY, "c1"), environment.state)
    assert result.status == "success" and result.verification.hard_valid
    assert result.verification.factual_support == pytest.approx(1 / (1 + math.e))
    assert result.verification.visual_support is None
    assert len(backend.calls) == 1
    assert result.verification.specialist_version == verifier.version


def test_verifier_checks_provenance_before_model_and_preserves_conflicts():
    environment = env()
    environment.add_candidate(Candidate("bad", "panda", "family", "ursidae", ()))
    backend = Backend(logits=[])
    result = agents.CalibratedVerificationAgent(backend, environment.schema, 1, 1)(
        Action(Operation.VERIFY, "bad"), environment.state
    )
    assert not result.verification.hard_valid and not backend.calls


def test_verifier_separately_scores_images_and_calibrates_visual_temperature(tmp_path):
    source = image_source(tmp_path)
    environment = env([source])
    result = retrieved(environment, Operation.RETRIEVE_IMAGE)
    environment.add_candidate(
        Candidate("v", "panda", "color", "red", result.anchors, modality="visual")
    )
    backend = Backend(logits=[2, -4])
    result = agents.CalibratedVerificationAgent(backend, environment.schema, 2, 4)(
        Action(Operation.VERIFY, "v"), environment.state
    )
    verification = result.verification
    assert verification.factual_support == pytest.approx(1 / (1 + math.exp(-1)))
    assert verification.visual_support == pytest.approx(1 / (1 + math.exp(1)))
    assert backend.calls[0][1] == () and backend.calls[1][1] == (source.path,)
    assert verification.uncertainty > 0.8


def test_dispatcher_versions_include_prompts_calibration_and_deterministic_gia():
    environment = env()
    backend = Backend()
    pea = agents.EvidenceAgent()
    tca = agents.TextualCandidateAgent(backend, environment.schema)
    vga = agents.VisualGroundingAgent(backend, environment.schema)
    verifier = agents.CalibratedVerificationAgent(backend, environment.schema, 1, 1)
    fixed = agents.FixedSpecialists(pea, tca, vga, verifier)
    assert set(fixed.versions) == {op.value for op in Operation}
    assert (
        fixed.versions["verify"]
        != agents.CalibratedVerificationAgent(backend, environment.schema, 2, 1).version
    )
    assert tca.version != vga.version
    assert fixed(Action(Operation.STOP), environment.state).status == "success"
    assert not backend.calls


@pytest.mark.parametrize("change", [{"quote": None}, {"quote": 2}])
def test_text_support_requires_explicit_exact_string_quote(change):
    environment = env()
    retrieved(environment)
    data = output()
    data["candidates"][0]["support"][0].update(change)
    result = agents.TextualCandidateAgent(Backend(data), environment.schema)(
        Action(Operation.GENERATE_TEXT, "e1"), environment.state
    )
    assert result.status == "parse_error" and not result.candidates


@pytest.mark.parametrize(
    "endpoint", [{"name": None, "entity_type": "family"}, {"name": 4, "entity_type": "family"}]
)
def test_malformed_typed_endpoints_are_atomic_parse_errors(endpoint):
    environment = env()
    retrieved(environment)
    result = agents.TextualCandidateAgent(Backend(output(tail=endpoint)), environment.schema)(
        Action(Operation.GENERATE_TEXT, "e1"), environment.state
    )
    assert result.status == "parse_error" and not result.candidates


def test_pea_checks_max_pixels_before_decoding(tmp_path, monkeypatch):
    source = image_source(tmp_path)
    monkeypatch.setattr(
        Image.Image, "load", lambda self, *args, **kwargs: pytest.fail("Pixel decoder ran")
    )
    result = agents.EvidenceAgent(max_image_pixels=100)(
        Action(Operation.RETRIEVE_IMAGE, "panda"), env([source]).state
    )
    assert result.status == "invalid" and "max_image_pixels" in result.message


def test_verifier_raw_score_collection_uses_identical_prompts():
    environment = env()
    evidence = retrieved(environment)
    environment.add_candidate(Candidate("c1", "panda", "family", "ursidae", evidence.anchors))
    backend = Backend(logits=[2, 2])
    verifier = agents.CalibratedVerificationAgent(backend, environment.schema, 1, 1)
    raw = verifier.raw_scores(environment.state.candidates["c1"], environment.state)
    assert raw == {"factual_logit": 2, "visual_logit": None}
    verifier(Action(Operation.VERIFY, "c1"), environment.state)
    assert backend.calls[0] == backend.calls[1]


def test_verifier_conflict_increases_uncertainty_without_erasing_records():
    environment = env()
    evidence = retrieved(environment)
    environment.state.entities["other"] = Entity("other", "OtherFamily", "family")
    environment.add_candidate(Candidate("c1", "panda", "family", "ursidae", evidence.anchors))
    environment.add_candidate(Candidate("c2", "panda", "family", "other", evidence.anchors))
    before = dict(environment.state.conflicts)
    result = agents.CalibratedVerificationAgent(Backend(logits=[8]), environment.schema, 1, 1)(
        Action(Operation.VERIFY, "c1"), environment.state
    )
    assert result.verification.hard_valid
    assert "unresolved source conflict" in result.verification.issues
    assert result.verification.uncertainty > 0.25
    assert environment.state.conflicts == before


def test_gia_has_image_preflight_requires_matching_tail(tmp_path):
    environment = env([image_source(tmp_path)])
    result = retrieved(environment, Operation.RETRIEVE_IMAGE)
    environment.add_candidate(
        Candidate("v", "panda", "hasImage", "invented", result.anchors, modality="visual")
    )
    assert (
        agents.GraphIntegrationAgent(environment.schema)(
            Action(Operation.ACCEPT, "v"), environment.state
        ).status
        == "invalid"
    )


def test_textual_prompt_always_includes_original_text_when_anchor_quote_is_absent():
    environment = env()
    retrieved(environment)
    environment.state.evidence[0] = replace(
        environment.state.evidence[0],
        anchors=(replace(environment.state.evidence[0].anchors[0], quote=None),),
    )
    backend = Backend(output())
    agents.TextualCandidateAgent(backend, environment.schema)(
        Action(Operation.GENERATE_TEXT, "e1"), environment.state
    )
    assert text_source().text == json.loads(backend.calls[0][1])["evidence"][0]["quote"]


def test_generator_rejects_unrelated_active_head_for_target_evidence():
    environment = env()
    retrieved(environment)
    environment.state.entities["other"] = Entity("other", "Other", "animal")
    environment.state.active_entity_ids.add("other")
    result = agents.TextualCandidateAgent(Backend(output(head="Other")), environment.schema)(
        Action(Operation.GENERATE_TEXT, "e1"), environment.state
    )
    assert result.status == "parse_error" and not result.candidates


def test_actual_dispatcher_pipeline_accepts_typed_new_entity_with_provenance():
    environment = env()
    backend = Backend(output(tail={"name": "NewFamily", "entity_type": "family"}), logits=[-3])
    fixed = agents.FixedSpecialists(
        agents.EvidenceAgent(),
        agents.TextualCandidateAgent(backend, environment.schema),
        agents.VisualGroundingAgent(backend, environment.schema),
        agents.CalibratedVerificationAgent(backend, environment.schema, 1, 1),
    )
    environment.dispatcher, environment.specialist_versions = fixed, fixed.versions
    environment.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    evidence_id = environment.state.evidence[-1].evidence_id
    environment.step(Action(Operation.GENERATE_TEXT, evidence_id))
    assert environment.state.evidence[-1].status == "success"
    candidate = next(iter(environment.state.candidates.values()))
    assert environment.state.entities[candidate.tail].entity_type == "family"
    environment.step(Action(Operation.VERIFY, candidate.candidate_id))
    environment.step(Action(Operation.ACCEPT, candidate.candidate_id))
    assert ("panda", "family", candidate.tail) in environment.state.graph.triples()


def test_verifier_calibration_score_version_binds_prompts_but_excludes_temperatures(monkeypatch):
    from rl_mac_mkg.agents import verification

    environment, backend = env(), Backend()
    first = agents.CalibratedVerificationAgent(backend, environment.schema, 1, 1)
    other_temperature = agents.CalibratedVerificationAgent(backend, environment.schema, 2, 1)
    assert first.score_version == other_temperature.score_version
    assert first.version != other_temperature.version
    monkeypatch.setattr(verification, "FACTUAL_PROMPT", verification.FACTUAL_PROMPT + " changed")
    changed = agents.CalibratedVerificationAgent(backend, environment.schema, 1, 1)
    assert changed.score_version != first.score_version
    schema = Schema({"family": PredicateRule(("animal",), ("family",), False)})
    assert (
        agents.CalibratedVerificationAgent(backend, schema, 1, 1).score_version
        != changed.score_version
    )
