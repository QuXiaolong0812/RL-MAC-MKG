from dataclasses import FrozenInstanceError, replace
from hashlib import sha256

import pytest

from rl_mac_mkg.environment import Environment, InvalidAction, RewardConfig
from rl_mac_mkg.graph import AliasCanonicalizer, PredicateRule, Schema
from rl_mac_mkg.records import (
    Action,
    Candidate,
    CandidateStatus,
    Entity,
    Operation,
    SourceAnchor,
    SourceRecord,
    SpecialistResult,
    Verification,
)

TEXT = "Panda 熊猫\r\nAiluropoda melanoleuca belongs to Ursidae."


def source(source_id="s1", text=TEXT):
    return SourceRecord(
        source_id,
        ("panda",),
        "https://example.org/" + source_id,
        "2026-10-19T10:00:00Z",
        "CC-BY-4.0",
        sha256(text.encode("utf-8")).hexdigest(),
        "text",
        text=text,
        document_id=source_id,
    )


def anchor(record=None, start=0, end=None):
    record = record or source()
    return SourceAnchor(
        record.source_id,
        record.url,
        record.retrieved_at,
        record.content_hash,
        document_id=record.document_id,
        start=start,
        end=end or len(record.text),
    )


def config():
    return RewardConfig(
        lambda_h=1,
        lambda_p=0.5,
        lambda_g=0.2,
        costs={o: (0 if o is Operation.STOP else 0.1) for o in Operation},
    )


def candidate(candidate_id="c1", tail="ursidae", anchors=None, **kwargs):
    return Candidate(
        candidate_id, "panda", "family", tail, (anchor(),) if anchors is None else anchors, **kwargs
    )


def environment(dispatcher=None, **kwargs):
    record = source()
    if kwargs.get("partition", "inference") != "inference":
        record = replace(record, partition=kwargs["partition"])
    return Environment(
        [
            Entity("panda", "Panda", "animal"),
            Entity("ursidae", "Ursidae", "family"),
            Entity("other", "Other", "family"),
        ],
        [record],
        Schema({"family": PredicateRule(("animal",), ("family",), single_valued=True)}),
        config(),
        dispatcher=dispatcher,
        **kwargs,
    )


def add_candidate(env, value=None):
    env.add_candidate(value or candidate())


def test_anchor_validates_original_unicode_and_newline_offsets():
    record = source()
    assert anchor(record, 6, 8).validate({record.source_id: record}) == "熊猫"
    with pytest.raises(ValueError, match="span"):
        replace(anchor(record), end=len(TEXT) + 1).validate({"s1": record})
    with pytest.raises(ValueError, match="hash"):
        replace(anchor(record), content_hash="0" * 64).validate({"s1": record})
    with pytest.raises(FrozenInstanceError):
        record.text = "tampered"
    with pytest.raises(ValueError, match="hash"):
        replace(record, text=record.text.replace("\r\n", "\n"))


def test_reward_configuration_requires_all_explicit_nonstop_costs():
    with pytest.raises(ValueError, match="cost"):
        RewardConfig(1, 1, 1, costs={Operation.STOP: 0})
    with pytest.raises(ValueError, match="cost"):
        replace(config(), costs={o: 0 for o in Operation})


def test_stop_at_zero_budget_and_serialization_are_deterministic():
    env = environment(budget=0)
    assert env.available_actions() == [Action(Operation.STOP)]
    initial_hash = env.state.hash()
    assert initial_hash == env.state.hash()
    transition = env.step(Action(Operation.STOP))
    assert transition.done and transition.reward == 0
    assert transition.old_state_hash == initial_hash
    assert transition.new_state_hash == env.state.hash()
    assert env.state.to_dict()["budget"] == 0


def test_accept_mask_cannot_be_bypassed_without_provenance_or_schema():
    env = environment()
    add_candidate(env, candidate(anchors=()))
    action = Action(Operation.ACCEPT, "c1")
    assert action not in env.available_actions()
    with pytest.raises(InvalidAction, match="provenance"):
        env.step(action)
    assert not env.state.graph.assertions
    assert env.state.evidence[-1].status == "invalid"
    assert env.state.budget == pytest.approx(29.9)


def test_low_soft_support_is_not_a_hard_mask_and_reward_cannot_repeat():
    env = environment()
    add_candidate(env)
    env.record_verification(Verification("c1", 1, 0.1, uncertainty=0.8))
    action = Action(Operation.ACCEPT, "c1")
    assert action in env.available_actions()
    result = env.step(action)
    assert result.reward_components["hard"] == 1
    assert result.reward_components["proxy"] == pytest.approx(0.1 - 0.2 * 0.8)
    assert env.state.candidates["c1"].status is CandidateStatus.ACCEPTED
    with pytest.raises(InvalidAction):
        env.step(action)
    assert len(env.state.graph.assertions) == 1


def test_failed_empty_retrieval_consumes_cost_and_blocks_identical_retry():
    calls = []

    def dispatch(action, state):
        calls.append(action)
        return SpecialistResult("empty", message="No source matched")

    env = environment(dispatch)
    action = Action(Operation.RETRIEVE_TEXT, "panda")
    transition = env.step(action)
    assert transition.reward_components["hard"] == -1
    assert env.state.evidence[-1].status == "empty"
    assert env.state.budget == pytest.approx(29.9)
    assert action not in env.available_actions()
    assert len(calls) == 1


def test_incremental_retrieval_cursor_and_generation_preserve_depth():
    seen = []

    def dispatch(action, state):
        cursor = state.retrieval_cursors.get("panda:text", 0)
        seen.append(cursor)
        return SpecialistResult("success", anchors=(anchor(start=cursor * 3, end=cursor * 3 + 3),))

    env = environment(dispatch)
    action = Action(Operation.RETRIEVE_TEXT, "panda")
    env.step(action)
    assert action in env.available_actions()
    env.step(action)
    assert seen == [0, 1]
    assert env.state.depth == 0


def test_revise_creates_version_and_invalidates_stale_verification():
    def dispatch(action, state):
        return SpecialistResult("success", candidates=(candidate(tail="other"),))

    env = environment(dispatch)
    add_candidate(env)
    env.record_verification(Verification("c1", 1, 0.9))
    env.step(Action(Operation.REVISE, "c1"))
    assert env.state.candidates["c1"].version == 2
    assert env.state.candidate_history["c1"][0].status is CandidateStatus.REVISED
    assert "c1" not in env.state.verifications
    with pytest.raises(ValueError, match="version"):
        env.record_verification(Verification("c1", 1, 0.99))


def test_conflicting_sources_are_retained_and_single_value_cannot_overwrite():
    env = environment()
    add_candidate(env)
    env.step(Action(Operation.ACCEPT, "c1"))
    add_candidate(env, candidate("c2", tail="other"))
    assert env.state.conflicts
    assert Action(Operation.ACCEPT, "c2") not in env.available_actions()
    with pytest.raises(InvalidAction):
        env.step(Action(Operation.ACCEPT, "c2"))
    assert next(iter(env.state.graph.assertions.values())).tail == "ursidae"


def test_duplicate_aliases_and_bad_predicates_are_hard_constraints():
    env = environment(canonicalizer=AliasCanonicalizer({"bear family": "ursidae"}))
    add_candidate(env)
    env.step(Action(Operation.ACCEPT, "c1"))
    add_candidate(env, candidate("c2", tail="bear family"))
    add_candidate(env, replace(candidate("c3"), predicate="unknown"))
    assert Action(Operation.ACCEPT, "c2") not in env.available_actions()
    assert Action(Operation.ACCEPT, "c3") not in env.available_actions()


def test_expand_alone_changes_depth_and_maximum_hops_blocks_it():
    env = environment(max_depth=1)
    add_candidate(env)
    env.step(Action(Operation.ACCEPT, "c1"))
    frontier_id = next(f.frontier_id for f in env.state.frontier.values() if f.entity_id == "panda")
    env.step(Action(Operation.EXPAND, frontier_id))
    assert env.state.depth == 1
    assert all(a.operation is not Operation.EXPAND for a in env.available_actions())


def test_training_annotations_are_never_visible_and_cannot_enter_test_rewards():
    with pytest.raises(ValueError, match="PPO"):
        environment(partition="test", reference_graph={("panda", "family", "ursidae")})
    env = environment(partition="ppo", reference_graph={("panda", "family", "ursidae")})
    add_candidate(env)
    env.step(Action(Operation.ACCEPT, "c1"))
    env.step(Action(Operation.STOP))
    assert env.terminal_reward() == pytest.approx(0.6)
    assert "reference_graph" not in env.state.to_dict()


def test_snapshot_roundtrip_preserves_hash_and_set_semantics():
    from rl_mac_mkg.environment import ConstructionState

    env = environment(lambda a, s: SpecialistResult("empty"))
    env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    snapshot = ConstructionState.from_dict(env.state.to_dict())
    assert snapshot.hash() == env.state.hash()
    assert isinstance(snapshot.call_fingerprints, set)
    assert snapshot.entities["panda"].name == "Panda"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: replace(source(), eligible="yes"),
        lambda: replace(source(), entity_ids="panda"),
        lambda: candidate(anchors=("forged",)),
        lambda: candidate(parse_valid="true"),
        lambda: Verification("c1", 1, True),
        lambda: SpecialistResult("success", candidates=("bad",)),
    ],
)
def test_typed_records_reject_ambiguous_or_untyped_values(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()


def test_dispatcher_mutation_cannot_change_graph_or_budget():
    def dispatch(action, state):
        state.budget = 99999
        state.graph.assertions["untrusted"] = "malformed"
        return SpecialistResult("empty")

    env = environment(dispatch)
    env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    assert env.state.budget == pytest.approx(29.9)
    assert not env.state.graph.assertions


def test_failure_is_atomic_and_records_timeout_without_retry():
    calls = []

    def dispatch(action, state):
        calls.append(action)
        raise TimeoutError("deadline")

    env = environment(dispatch)
    transition = env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    assert transition.reward_components["hard"] == -1
    assert env.state.evidence[-1].status == "timeout"
    assert len(calls) == 1


def test_partial_generation_failure_cannot_insert_earlier_candidates():
    def dispatch(action, state):
        if action.operation is Operation.RETRIEVE_TEXT:
            return SpecialistResult("success", anchors=(anchor(),))
        return SpecialistResult("success", candidates=(candidate(), candidate("bad", version=2)))

    env = environment(dispatch)
    env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    result = env.step(Action(Operation.GENERATE_TEXT, env.state.evidence[0].evidence_id))
    assert not env.state.candidates
    assert result.reward_components["hard"] == -1


def test_reject_resolves_conflict_but_keeps_full_record():
    env = environment()
    add_candidate(env)
    add_candidate(env, candidate("c2", tail="other"))
    assert any(not c.resolved for c in env.state.conflicts.values())
    env.step(Action(Operation.REJECT, "c2"))
    assert env.state.conflicts
    assert all(c.resolved for c in env.state.conflicts.values())


def test_terminal_contradiction_only_counts_constructed_graph():
    env = environment(partition="ppo", reference_graph={("panda", "family", "ursidae")})
    add_candidate(env)
    env.step(Action(Operation.ACCEPT, "c1"))
    add_candidate(env, candidate("c2", tail="other"))
    env.step(Action(Operation.STOP))
    assert env.terminal_reward() == pytest.approx(0.6)


def test_related_new_evidence_unblocks_failed_generation():
    def dispatch(action, state):
        if action.operation is Operation.RETRIEVE_TEXT:
            start = state.retrieval_cursors.get("panda:text", 0)
            return SpecialistResult("success", anchors=(anchor(start=start, end=start + 3),))
        return SpecialistResult("parse_error", message="invalid JSON")

    env = environment(dispatch)
    env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    generation = Action(Operation.GENERATE_TEXT, env.state.evidence[0].evidence_id)
    env.step(generation)
    assert generation not in env.available_actions()
    env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    assert generation in env.available_actions()


def test_noninference_episodes_require_explicit_matching_source_partitions():
    with pytest.raises(ValueError, match="partition"):
        Environment(
            [Entity("panda", "Panda", "animal")],
            [source()],
            Schema({"family": PredicateRule(("animal",), ("family",))}),
            config(),
            partition="ppo",
        )


def test_untrusted_specialist_cannot_substitute_another_entity_source():
    record = source()
    env = environment(
        lambda a, s: SpecialistResult("success", anchors=(anchor(record),)),
        seed_entity_ids=("panda", "other"),
    )
    transition = env.step(Action(Operation.RETRIEVE_TEXT, "other"))
    assert transition.reward_components["hard"] == -1
    assert env.state.evidence[-1].status == "invalid"
    assert not env.state.retrieval_cursors


def test_explicit_specialist_version_mismatch_is_a_recorded_failure():
    env = environment(
        lambda a, s: SpecialistResult("success", anchors=(anchor(),), specialist_version="old"),
        specialist_versions={"retrieve_text": "new"},
    )
    result = env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    assert result.reward_components["hard"] == -1
    assert "version" in env.state.evidence[-1].message
    assert env.state.evidence[-1].specialist_version == "old"


def test_reward_uncertainty_penalty_is_explicitly_configurable():
    env = environment()
    env.reward_config = replace(config(), lambda_u=0.8)
    add_candidate(env)
    env.record_verification(Verification("c1", 1, 0.5, uncertainty=0.5))
    result = env.step(Action(Operation.ACCEPT, "c1"))
    assert result.reward_components["proxy"] == pytest.approx(0.1)


def test_catalog_tail_entities_are_unavailable_until_expand():
    env = environment(max_depth=2)
    action = Action(Operation.RETRIEVE_TEXT, "ursidae")
    assert action not in env.available_actions()
    add_candidate(env)
    env.step(Action(Operation.ACCEPT, "c1"))
    assert action not in env.available_actions()
    env.step(Action(Operation.EXPAND, "frontier:panda:0"))
    assert action in env.available_actions()


def test_hop_zero_still_allows_seed_evidence_to_generate_candidates():
    env = environment(lambda a, s: SpecialistResult("success", anchors=(anchor(),)), max_depth=0)
    transition = env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    assert not transition.done
    assert Action(Operation.GENERATE_TEXT, "evidence:000001") in env.available_actions()


def test_default_canonicalizer_maps_names_and_explicit_aliases():
    env = environment()
    add_candidate(env, replace(candidate(), head="Panda", tail="Ursidae"))
    env.step(Action(Operation.ACCEPT, "c1"))
    assert env.state.graph.triples() == {("panda", "family", "ursidae")}


def test_hasimage_requires_the_matching_image_anchor():
    record = SourceRecord(
        "img1",
        ("panda",),
        "https://example.org/img",
        "2026-10-19T10:00:00Z",
        "CC0",
        "1" * 64,
        "image",
        image_id="i1",
    )
    image_anchor = SourceAnchor(
        "img1", record.url, record.retrieved_at, record.content_hash, image_id="i1"
    )
    env = Environment(
        [Entity("panda", "Panda", "animal")],
        [record],
        Schema({"hasImage": PredicateRule(("animal",), tail_is_entity=False)}),
        config(),
    )
    add_candidate(
        env, Candidate("c1", "panda", "hasImage", "i2", (image_anchor,), modality="visual")
    )
    assert Action(Operation.ACCEPT, "c1") not in env.available_actions()


def test_revision_keeps_prior_verifier_output_in_evidence_history():
    def dispatch(action, state):
        if action.operation is Operation.VERIFY:
            return SpecialistResult("success", verification=Verification("c1", 1, 0.9))
        return SpecialistResult("success", candidates=(candidate(tail="other"),))

    env = environment(dispatch)
    add_candidate(env)
    env.step(Action(Operation.VERIFY, "c1"))
    env.step(Action(Operation.REVISE, "c1"))
    assert env.state.evidence[0].verification.candidate_version == 1
    assert "c1" not in env.state.verifications


def test_generation_incorporates_new_typed_tail_but_only_expand_activates_it():
    new_entity = Entity("ailuropodinae", "Ailuropodinae", "family", ("Panda subfamily",))
    fixed = AliasCanonicalizer({"Bear family": "ursidae"})
    original_aliases = dict(fixed.aliases)

    def dispatch(action, state):
        if action.operation is Operation.RETRIEVE_TEXT:
            return SpecialistResult("success", anchors=(anchor(),))
        return SpecialistResult(
            "success", candidates=(candidate(tail="Ailuropodinae"),), entities=(new_entity,)
        )

    env = environment(dispatch, canonicalizer=fixed, max_depth=2)
    env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    env.step(Action(Operation.GENERATE_TEXT, "evidence:000001"))
    assert env.state.entities["ailuropodinae"] == new_entity
    assert env.canonicalizer("Panda subfamily") == "ailuropodinae"
    assert Action(Operation.RETRIEVE_TEXT, "ailuropodinae") not in env.available_actions()
    env.step(Action(Operation.ACCEPT, "c1"))
    env.step(Action(Operation.EXPAND, "frontier:panda:0"))
    assert Action(Operation.RETRIEVE_TEXT, "ailuropodinae") in env.available_actions()
    assert fixed.aliases == original_aliases


@pytest.mark.parametrize(
    "conflicting",
    [
        Entity("panda", "Changed panda", "animal"),
        Entity("panda", "Panda", "family"),
        Entity("new_id", "Panda", "family"),
        Entity("new_id", "New family", "family", ("ursidae",)),
    ],
)
def test_conflicting_entity_proposals_leave_candidate_and_entity_stores_atomic(conflicting):
    def dispatch(action, state):
        if action.operation is Operation.RETRIEVE_TEXT:
            return SpecialistResult("success", anchors=(anchor(),))
        return SpecialistResult(
            "success",
            candidates=(candidate(tail="new_family"),),
            entities=(Entity("new_family", "New family", "family"), conflicting),
        )

    env = environment(dispatch)
    env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    before = dict(env.state.entities)
    transition = env.step(Action(Operation.GENERATE_TEXT, "evidence:000001"))
    assert transition.reward_components["hard"] == -1
    assert env.state.entities == before
    assert not env.state.candidates


def test_revise_can_introduce_new_tail_and_keeps_it_inactive():
    def dispatch(action, state):
        return SpecialistResult(
            "success",
            candidates=(candidate(tail="new_family"),),
            entities=(Entity("new_family", "New family", "family"),),
        )

    env = environment(dispatch)
    add_candidate(env)
    env.step(Action(Operation.REVISE, "c1"))
    assert env.state.candidates["c1"].version == 2
    assert env.state.entities["new_family"].entity_type == "family"
    assert "new_family" not in env.state.active_entity_ids


def test_generation_cannot_use_its_own_ledger_as_another_generation_target():
    def dispatch(action, state):
        if action.operation is Operation.RETRIEVE_TEXT:
            return SpecialistResult("success", anchors=(anchor(),))
        return SpecialistResult("success", anchors=(anchor(),), candidates=(candidate(),))

    env = environment(dispatch)
    env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    env.step(Action(Operation.GENERATE_TEXT, "evidence:000001"))
    assert Action(Operation.GENERATE_TEXT, "evidence:000002") not in env.available_actions()


@pytest.mark.parametrize(
    "tail,proposed",
    [
        ("missing", ()),
        ("bad_type", (Entity("bad_type", "Bad family", "animal"),)),
    ],
)
def test_generated_endpoints_must_exist_with_schema_compatible_types(tail, proposed):
    def dispatch(action, state):
        if action.operation is Operation.RETRIEVE_TEXT:
            return SpecialistResult("success", anchors=(anchor(),))
        return SpecialistResult("success", candidates=(candidate(tail=tail),), entities=proposed)

    env = environment(dispatch)
    env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    transition = env.step(Action(Operation.GENERATE_TEXT, "evidence:000001"))
    assert transition.reward_components["hard"] == -1
    assert tail not in env.state.entities
    assert not env.state.candidates


def test_proposals_from_retrieval_are_rejected_and_do_not_create_entities():
    env = environment(
        lambda a, s: SpecialistResult(
            "success", anchors=(anchor(),), entities=(Entity("new_family", "New family", "family"),)
        )
    )
    transition = env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    assert transition.reward_components["hard"] == -1
    assert "new_family" not in env.state.entities


def test_generated_candidate_cannot_switch_to_an_inactive_catalog_head():
    def dispatch(action, state):
        if action.operation is Operation.RETRIEVE_TEXT:
            return SpecialistResult("success", anchors=(anchor(),))
        return SpecialistResult("success", candidates=(replace(candidate(), head="other"),))

    env = environment(dispatch)
    env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    transition = env.step(Action(Operation.GENERATE_TEXT, "evidence:000001"))
    assert transition.reward_components["hard"] == -1
    assert not env.state.candidates


def test_generated_anchors_cannot_come_from_an_unacquired_catalog_source():
    unrelated = source("s2")

    def dispatch(action, state):
        if action.operation is Operation.RETRIEVE_TEXT:
            return SpecialistResult("success", anchors=(anchor(),))
        return SpecialistResult("success", candidates=(candidate(anchors=(anchor(unrelated),)),))

    env = environment(dispatch)
    env.state.sources["s2"] = unrelated
    env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    transition = env.step(Action(Operation.GENERATE_TEXT, "evidence:000001"))
    assert transition.reward_components["hard"] == -1
    assert not env.state.candidates


def test_duplicate_retrieval_cannot_advance_cursor_or_bypass_repeat_mask():
    env = environment(lambda a, s: SpecialistResult("success", anchors=(anchor(),)))
    action = Action(Operation.RETRIEVE_TEXT, "panda")
    env.step(action)
    transition = env.step(action)
    assert transition.reward_components["hard"] == -1
    assert env.state.evidence[-1].status == "invalid"
    assert "duplicate" in env.state.evidence[-1].message.lower()
    assert env.state.retrieval_cursors["panda:text"] == 1
    assert env.state.budget == pytest.approx(29.8)
    assert action not in env.available_actions()


def test_entity_proposals_cannot_attach_new_entities_only_to_literal_values():
    env = Environment(
        [Entity("panda", "Panda", "animal")],
        [source()],
        Schema({"label": PredicateRule(("animal",), tail_is_entity=False)}),
        config(),
        dispatcher=lambda a, s: (
            SpecialistResult("success", anchors=(anchor(),))
            if a.operation is Operation.RETRIEVE_TEXT
            else SpecialistResult(
                "success",
                candidates=(Candidate("c1", "panda", "label", "new_id", (anchor(),)),),
                entities=(Entity("new_id", "New entity", "animal"),),
            )
        ),
    )
    env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    transition = env.step(Action(Operation.GENERATE_TEXT, "evidence:000001"))
    assert transition.reward_components["hard"] == -1
    assert "new_id" not in env.state.entities


def test_entity_aliases_never_rewrite_predicates_or_literal_values():
    env = Environment(
        [
            Entity("panda", "Panda", "animal"),
            Entity("color_id", "color", "concept"),
            Entity("red_id", "red", "color"),
        ],
        [source()],
        Schema({"color": PredicateRule(("animal",), tail_is_entity=False, single_valued=True)}),
        config(),
    )
    env.add_candidate(Candidate("c1", "panda", "color", "red", (anchor(),)))
    assert Action(Operation.ACCEPT, "c1") in env.available_actions()
    env.step(Action(Operation.ACCEPT, "c1"))
    assert env.state.graph.triples() == {("panda", "color", "red")}
    env.add_candidate(Candidate("c2", "panda", "color", "blue", (anchor(),)))
    conflict = next(iter(env.state.conflicts.values()))
    assert conflict.predicate == "color"
    assert conflict.values == ("blue", "red")
    env.step(Action(Operation.REJECT, "c2"))
    assert all(c.resolved for c in env.state.conflicts.values())


def test_image_identifiers_are_preserved_despite_matching_entity_aliases():
    record = SourceRecord(
        "img1",
        ("panda",),
        "https://example.org/img",
        "2026-10-19T10:00:00Z",
        "CC0",
        "1" * 64,
        "image",
        image_id="i1",
    )
    image_anchor = SourceAnchor(
        "img1", record.url, record.retrieved_at, record.content_hash, image_id="i1"
    )
    env = Environment(
        [Entity("panda", "Panda", "animal"), Entity("other_id", "i1", "concept")],
        [record],
        Schema({"hasImage": PredicateRule(("animal",), tail_is_entity=False)}),
        config(),
    )
    env.add_candidate(
        Candidate("c1", "panda", "hasImage", "i1", (image_anchor,), modality="visual")
    )
    env.step(Action(Operation.ACCEPT, "c1"))
    assert env.state.graph.triples() == {("panda", "hasImage", "i1")}


def test_unknown_source_anchor_is_recorded_as_invalid_without_crashing_masks():
    forged = replace(anchor(), source_id="unknown")
    env = environment(lambda a, s: SpecialistResult("success", anchors=(forged,)))
    transition = env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    assert transition.reward_components["hard"] == -1
    assert env.state.budget == pytest.approx(29.9)
    assert env.state.evidence[-1].status == "invalid"
    assert env.state.evidence[-1].anchors[0].source_id == "unknown"
    assert (
        Action(Operation.GENERATE_TEXT, env.state.evidence[-1].evidence_id)
        not in env.available_actions()
    )


@pytest.mark.parametrize("head,tail", [("panda", "ursidae"), ("ursidae", "panda")])
def test_literal_attributes_cannot_activate_catalog_entities_in_either_direction(head, tail):
    env = Environment(
        [Entity("panda", "Panda", "animal"), Entity("ursidae", "Ursidae", "animal")],
        [source()],
        Schema({"label": PredicateRule(("animal",), tail_is_entity=False)}),
        config(),
        max_depth=2,
    )
    env.add_candidate(Candidate("c1", head, "label", tail, (anchor(),)))
    env.step(Action(Operation.ACCEPT, "c1"))
    env.step(Action(Operation.EXPAND, "frontier:panda:0"))
    assert "ursidae" not in env.state.active_entity_ids
    assert all(f.entity_id != "ursidae" for f in env.state.frontier.values())


def test_real_chunk_retriever_remains_pending_after_first_resolved_chunk_at_hop_zero():
    from rl_mac_mkg.agents import EvidenceAgent

    pea = EvidenceAgent(chunk_chars=18, chunk_overlap=0)

    def dispatch(action, state):
        if action.operation is Operation.RETRIEVE_TEXT:
            return pea(action, state)
        evidence = next(e for e in state.evidence if e.evidence_id == action.target_id)
        return SpecialistResult(
            "success",
            candidates=(
                candidate(f"c{state.retrieval_cursors['panda:text']}", anchors=evidence.anchors),
            ),
        )

    env = environment(dispatch, max_depth=0)
    env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    first = env.state.evidence[-1]
    env.step(Action(Operation.GENERATE_TEXT, first.evidence_id))
    transition = env.step(Action(Operation.REJECT, "c1"))
    assert not transition.done
    env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    assert env.state.evidence[-1].status == "success"
    assert env.state.evidence[-1].anchors[0].start >= first.anchors[0].end


def test_retrieval_exhaustion_is_explicit_and_survives_snapshot_roundtrip():
    from rl_mac_mkg.agents import EvidenceAgent
    from rl_mac_mkg.environment import ConstructionState

    pea = EvidenceAgent(chunk_chars=18, chunk_overlap=0)
    env = environment(pea)
    action = Action(Operation.RETRIEVE_TEXT, "panda")
    for _ in range(20):
        env.step(action)
        if env.state.evidence[-1].status == "empty":
            break
    else:
        pytest.fail("Finite source chunks must eventually report exhaustion")
    assert env.state.retrieval_exhausted == {"panda:text"}
    assert action not in env.available_actions()
    restored = ConstructionState.from_dict(env.state.to_dict())
    assert restored.retrieval_exhausted == {"panda:text"}
    assert restored.hash() == env.state.hash()


def test_deterministic_graph_ledger_records_configured_integration_version():
    env = environment(specialist_versions={"accept": "GIA:fixed-v1"})
    add_candidate(env)
    env.step(Action(Operation.ACCEPT, "c1"))
    assert env.state.evidence[-1].specialist_version == "GIA:fixed-v1"


def test_dispatch_timeout_ledger_records_attempted_configured_version():
    def timeout(action, state):
        raise TimeoutError("deadline")

    env = environment(timeout, specialist_versions={"retrieve_text": "PEA:fixed-v1"})
    env.step(Action(Operation.RETRIEVE_TEXT, "panda"))
    assert env.state.evidence[-1].status == "timeout"
    assert env.state.evidence[-1].specialist_version == "PEA:fixed-v1"
