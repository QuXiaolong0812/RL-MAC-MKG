"""Deterministic graph integration preflight; Environment owns atomic graph mutations."""

from dataclasses import replace

from ..graph import AliasCanonicalizer
from ..records import Operation, SpecialistResult
from .common import Specialist, version


class GraphIntegrationAgent(Specialist):
    def __init__(self, schema=None, *, canonicalizer=None):
        self.schema, self.canonicalizer = schema, canonicalizer
        self.version = version(
            "GIA",
            schema=schema,
            canonicalizer=getattr(canonicalizer, "version", None),
            contract_name="environment-atomic-accept-reject-expand-v1",
        )

    def execute(self, action, state):
        if action.operation is Operation.STOP:
            return SpecialistResult("success")
        if action.operation is Operation.EXPAND:
            item = state.frontier.get(action.target_id)
            if item is None or item.depth >= state.max_depth:
                raise ValueError("Expansion requires an active frontier item below maximum depth")
            return SpecialistResult("success")
        if action.operation not in (Operation.ACCEPT, Operation.REJECT):
            raise ValueError("GIA accepts deterministic graph operations only")
        candidate = state.candidates[action.target_id]
        if not candidate.unresolved:
            raise ValueError("Integration requires an unresolved candidate")
        if action.operation is Operation.ACCEPT:
            if self.schema is not None:
                issues = state.graph.issues(
                    candidate,
                    self.schema,
                    state.entities,
                    self.canonicalizer or AliasCanonicalizer(),
                )
                if issues:
                    raise ValueError("; ".join(issues))
            if not candidate.anchors:
                raise ValueError("Integration requires provenance")
            for anchor in candidate.anchors:
                anchor.validate(state.sources)
            if candidate.predicate in ("hasImage", "has_image") and candidate.tail not in {
                a.image_id for a in candidate.anchors if a.image_id is not None
            }:
                raise ValueError("hasImage requires a matching supplied image anchor")
            verification = state.verifications.get(candidate.candidate_id)
            if verification and (
                not verification.hard_valid or verification.candidate_version != candidate.version
            ):
                raise ValueError("Integration requires current hard-valid verification")
        return SpecialistResult("success", anchors=candidate.anchors)


class FixedSpecialists:
    def __init__(self, evidence, textual, visual, verifier, integration=None):
        self.evidence, self.textual, self.visual, self.verifier = (
            evidence,
            textual,
            visual,
            verifier,
        )
        self.canonicalizer = textual.canonicalizer
        self.integration = integration or GraphIntegrationAgent(
            textual.schema, canonicalizer=textual.canonicalizer
        )
        self.revision_version = version("REVISE", text=textual.version, visual=visual.version)
        self.versions = {op.value: self.integration.version for op in Operation}
        for operation, specialist in (
            (Operation.RETRIEVE_TEXT, evidence),
            (Operation.RETRIEVE_IMAGE, evidence),
            (Operation.GENERATE_TEXT, textual),
            (Operation.GENERATE_VISUAL, visual),
            (Operation.VERIFY, verifier),
        ):
            self.versions[operation.value] = specialist.version
        self.versions[Operation.REVISE.value] = self.revision_version

    def __call__(self, action, state):
        operation = action.operation
        if operation in (Operation.RETRIEVE_TEXT, Operation.RETRIEVE_IMAGE):
            specialist = self.evidence
        elif operation is Operation.GENERATE_TEXT:
            specialist = self.textual
        elif operation is Operation.GENERATE_VISUAL:
            specialist = self.visual
        elif operation is Operation.VERIFY:
            specialist = self.verifier
        elif operation is Operation.REVISE:
            candidate = state.candidates[action.target_id]
            specialist = self.textual if candidate.modality == "text" else self.visual
        else:
            specialist = self.integration
        result = specialist(action, state)
        if operation is Operation.REVISE:
            result = replace(result, specialist_version=self.revision_version)
        return result
