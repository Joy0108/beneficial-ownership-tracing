from __future__ import annotations

import pytest

from ubo.config import SECRECY_JURISDICTIONS
from ubo.drift import population_stability_index, run_monitors, simulate_designation_round
from ubo.er.resolve import ResolvedEntity
from ubo.graph.build import effective_ownership
from ubo.graph.features import compute_features, describe
from ubo.graph.patterns import assess, evaluate, sweep_threshold
from ubo.rag import pep
from ubo.registers.loaders import Statement
from ubo.registry import Registry
from ubo.workflow.graph import END, START, StateMachine, WorkflowError
from ubo.workflow.nodes import record_decision, screen

# --- graph construction ----------------------------------------------------

def _entity(eid, name, jur, kind="company"):
    return ResolvedEntity(entity_id=eid, entity_type=kind, canonical_name=name,
                          record_ids=[eid], sources=["test"], jurisdictions=[jur])


def _chain_graph():
    entities = [
        _entity("P", "Owner", "RU", "person"),
        _entity("A", "Alpha Ltd", "VG"),
        _entity("B", "Beta Ltd", "KY"),
        _entity("C", "Gamma Ltd", "GB"),
    ]
    statements = [
        Statement("s1", "icij", "A", "P", "shareholding", 100.0, "2016-05-09"),
        Statement("s2", "icij", "B", "A", "shareholding", 50.0, "2016-05-09"),
        Statement("s3", "psc", "C", "B", "shareholding", 60.0, "2024-06-30"),
    ]
    from ubo.graph.build import build_graph

    return build_graph(entities, statements, {e.entity_id: e.entity_id for e in entities})


def test_edge_direction_follows_control():
    graph = _chain_graph()
    assert [e.child for e in graph.children("P")] == ["A"]
    assert graph.roots() == ["P"]


def test_ownership_attenuates_along_the_chain():
    graph = _chain_graph()
    effective = effective_ownership(graph, "P")
    # 100% of A, A holds 50% of B, B holds 60% of C -> 30% of C, not 60%.
    assert effective["B"] == pytest.approx(50.0)
    assert effective["C"] == pytest.approx(30.0)


def test_a_statement_whose_endpoints_do_not_resolve_is_recorded_not_dropped():
    from ubo.graph.build import build_graph

    entities = [_entity("A", "Alpha", "GB")]
    statements = [Statement("s1", "psc", "A", "MISSING", "shareholding", 50.0, "2024-01-01")]
    graph = build_graph(entities, statements, {"A": "A"})
    assert graph.edges == []
    assert len(graph.unresolved_statements) == 1


def test_corroborating_statements_merge_into_one_edge_keeping_both():
    from ubo.graph.build import build_graph

    entities = [_entity("A", "Alpha", "GB"), _entity("B", "Beta", "GB")]
    statements = [
        Statement("s1", "psc", "B", "A", "shareholding", 60.0, "2024-01-01", confidence=0.9),
        Statement("s2", "gleif_l2", "B", "A", "shareholding", 60.0, "2024-02-01", confidence=1.0),
    ]
    graph = build_graph(entities, statements, {"A": "A", "B": "B"})
    assert len(graph.edges) == 1
    assert len(graph.edges[0].statements) == 2
    assert graph.edges[0].confidence == pytest.approx(1.0)


def test_edge_confidence_carries_resolution_confidence_forward():
    from ubo.graph.build import build_graph

    weak = _entity("A", "Alpha", "GB")
    weak.weakest_link = 0.6
    entities = [weak, _entity("B", "Beta", "GB")]
    statements = [Statement("s1", "psc", "B", "A", "shareholding", 60.0, "2024-01-01", confidence=1.0)]
    graph = build_graph(entities, statements, {"A": "A", "B": "B"})
    assert graph.edges[0].confidence == pytest.approx(0.6)


def test_circular_ownership_is_detected_and_traversal_still_terminates():
    from ubo.graph.build import build_graph

    entities = [_entity(x, x, "CY") for x in "ABC"]
    statements = [
        Statement("s1", "t", "B", "A", "shareholding", 60.0, "2024-01-01"),
        Statement("s2", "t", "C", "B", "shareholding", 60.0, "2024-01-01"),
        Statement("s3", "t", "A", "C", "shareholding", 60.0, "2024-01-01"),
    ]
    graph = build_graph(entities, statements, {x: x for x in "ABC"})
    assert graph.cycles()
    assert set(graph.descendants("A")) == {"B", "C"}


# --- features and risk -----------------------------------------------------

def test_secrecy_hops_count_the_chain_not_the_corpus():
    graph = _chain_graph()
    feats = compute_features(graph, roots=["P"])["P"]
    assert feats.chain_depth == 3
    assert feats.secrecy_hops == 2  # VG and KY
    assert {"VG", "KY"} <= SECRECY_JURISDICTIONS


def test_ordinary_co_ownership_is_not_counted_as_a_nominee(graph, features, mapping):
    """The clean three-hop European structure must not be flagged."""
    clean = [eid for eid, seed in mapping.items() if seed == "P013" and eid in features]
    if not clean:
        pytest.skip("clean deep structure not present in this resolution")
    assert features[clean[0]].nominee_intermediaries == 0


def test_describe_only_reports_indicators_that_fired():
    graph = _chain_graph()
    feats = compute_features(graph, roots=["P"])["P"]
    reasons = describe(feats)
    assert any("secrecy" in r for r in reasons)
    assert not any("circular" in r for r in reasons)


def test_layering_detection_separates_the_labelled_structures(features, labels):
    result = evaluate({eid: assess(f) for eid, f in features.items()}, labels)
    assert result["precision"] >= 0.8, f"too many false flags: {result}"
    assert result["recall"] >= 0.6, f"missing real structures: {result}"


def test_threshold_sweep_is_monotonic_in_recall(features, labels):
    assessments = {eid: assess(f) for eid, f in features.items()}
    rows = sweep_threshold(assessments, labels)
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls, reverse=True)


# --- PEP language ----------------------------------------------------------

@pytest.mark.parametrize("text", [
    "The beneficial owner is a PEP, which is an adverse finding.",
    "The customer is politically exposed and the relationship must be declined.",
    "A PEP match indicates corruption on the part of the beneficial owner.",
])
def test_adverse_pep_language_is_rejected(text):
    assert not pep.check(text).passed


def test_correct_pep_framing_is_accepted():
    text = (
        "The beneficial owner is recorded as a PEP. Under Recommendation 12 this requires enhanced due "
        "diligence, senior management approval and source of wealth work. It is not an adverse finding and "
        "does not support declining the relationship."
    )
    assert pep.check(text).passed


def test_negated_adverse_terms_are_not_violations():
    assert pep.check("The PEP match is not an adverse finding and enhanced due diligence applies.").violations == []


def test_enforce_adds_the_framing_but_never_hides_a_violation():
    text, result = pep.enforce("The beneficial owner is a PEP.", is_pep=True)
    assert "enhanced due diligence" in text.lower()
    assert result.passed

    text, result = pep.enforce("The PEP is corrupt.", is_pep=True)
    assert result.violations, "an accusation must remain visible in the audit trail"


# --- workflow --------------------------------------------------------------

def test_state_machine_merges_partial_state_and_records_the_path():
    machine = StateMachine("t")
    machine.add_node("a", lambda s: {"x": 1})
    machine.add_node("b", lambda s: {"y": s["x"] + 1})
    machine.add_edge(START, "a")
    machine.add_edge("a", "b")
    machine.add_edge("b", END)
    out = machine.invoke({"seed": True})
    assert out["seed"] is True and out["y"] == 2
    assert out["_path"] == ["a", "b"]


def test_a_required_stage_that_is_routed_around_is_an_error():
    """The human gate is registered required so no routing change can skip it."""
    machine = StateMachine("t")
    machine.add_node("a", lambda s: {})
    machine.add_node("gate", lambda s: {}, required=True)
    machine.add_edge(START, "a")
    machine.add_edge("a", END)  # deliberately bypasses the gate
    with pytest.raises(WorkflowError, match="required stage"):
        machine.invoke({})


def test_the_engine_bounds_a_cycle():
    machine = StateMachine("t", max_steps=4)
    machine.add_node("a", lambda s: {"n": s.get("n", 0) + 1})
    machine.add_edge(START, "a")
    machine.add_conditional_edges("a", lambda s: "loop", {"loop": "a"})
    with pytest.raises(WorkflowError, match="max_steps"):
        machine.invoke({})


def test_screening_runs_all_eight_steps_and_stops_at_the_gate(records, statements, entities, graph, regulatory, mapping):
    subject = next(e for e in graph.entities if graph.children(e, control_only=True))
    state = screen(subject, state={"records": records, "statements": statements, "entities": entities},
                   regulatory=regulatory)
    assert state["_path"] == [
        "ingest", "resolve", "assemble_graph", "score_structure",
        "retrieve_guidance", "draft_memo", "verify", "human_gate",
    ]
    assert state["awaiting_human_decision"] is True
    assert state["decision_package"]["decision"] is None


def test_every_memo_citation_resolves(records, statements, entities, graph, regulatory):
    subject = next(e for e in graph.entities if graph.children(e, control_only=True))
    state = screen(subject, state={"records": records, "statements": statements, "entities": entities},
                   regulatory=regulatory)
    assert state["verification"]["citations"]["resolution_rate"] == 1.0
    assert state["verification"]["citations"]["unresolved"] == []


def test_a_memo_about_a_pep_carries_the_required_framing(records, statements, entities, graph, regulatory, mapping):
    pep_entities = [e for e in entities if e.is_pep and e.entity_id in graph.entities]
    if not pep_entities:
        pytest.skip("no PEP resolved in this run")
    state = screen(pep_entities[0].entity_id,
                   state={"records": records, "statements": statements, "entities": entities},
                   regulatory=regulatory)
    assert state["verification"]["pep"]["passed"]
    assert "enhanced due diligence" in state["memo"].text.lower()


def test_only_a_caller_with_an_identity_can_close_the_gate(records, statements, entities, graph, regulatory):
    subject = next(e for e in graph.entities if graph.children(e, control_only=True))
    state = screen(subject, state={"records": records, "statements": statements, "entities": entities},
                   regulatory=regulatory)
    closed = record_decision(state, "escalate", "analyst-1", "2024-08-20T10:00:00Z")
    assert closed["decision_package"]["decision"] == "escalate"
    assert closed["decision_package"]["decided_by"] == "analyst-1"
    assert closed["awaiting_human_decision"] is False
    with pytest.raises(ValueError):
        record_decision(state, "close_account", "analyst-1", "2024-08-20T10:00:00Z")


# --- registry --------------------------------------------------------------

def test_a_version_failing_a_blocking_gate_is_not_promoted(tmp_path):
    registry = Registry(path=tmp_path / "registry.json")
    version = registry.register({
        "er_f1": 0.99, "er_precision": 0.99, "candidate_recall": 0.99,
        "layering_precision": 0.95, "rag_recall_at_5": 0.95,
        "citation_resolution": 1.0, "pep_language_rate": 0.9,  # the absolute gate
    })
    result = registry.promote(version.version)
    assert result["promoted"] is False
    assert [g["gate"] for g in result["failed_gates"]] == ["pep_language"]


def test_a_clean_version_promotes_and_archives_the_previous_one(tmp_path):
    registry = Registry(path=tmp_path / "registry.json")
    good = {
        "er_f1": 0.96, "er_precision": 0.98, "candidate_recall": 0.99,
        "layering_precision": 1.0, "rag_recall_at_5": 0.96,
        "citation_resolution": 1.0, "pep_language_rate": 1.0,
    }
    first = registry.register(good)
    assert registry.promote(first.version)["promoted"]
    second = registry.register(good)
    assert registry.promote(second.version)["promoted"]
    assert registry.production().version == second.version
    assert next(v for v in registry.versions if v.version == first.version).stage == "archived"


def test_promotion_can_be_forced_but_the_failure_is_recorded(tmp_path):
    registry = Registry(path=tmp_path / "registry.json")
    version = registry.register({"er_f1": 0.5, "pep_language_rate": 1.0})
    result = registry.promote(version.version, force=True)
    assert result["promoted"] and result["forced"]
    assert result["failed_gates"]


# --- drift -----------------------------------------------------------------

def test_psi_is_zero_for_an_unchanged_distribution():
    from collections import Counter

    dist = Counter({"GB": 40, "CY": 30, "KY": 30})
    psi, _ = population_stability_index(dist, dist)
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_a_designation_round_trips_the_jurisdiction_monitor(records):
    shifted = simulate_designation_round(records, "RU", multiplier=8)
    report = run_monitors(records, shifted)
    jurisdiction = next(s for s in report["signals"] if s["signal"] == "jurisdiction_mix")
    assert jurisdiction["status"] in {"investigate", "alert"}
    assert report["overall_status"] != "stable"


def test_monitors_stay_quiet_when_nothing_moved(records):
    report = run_monitors(records, records)
    assert report["overall_status"] == "stable"
    assert report["action"] == "no action"
