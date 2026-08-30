"""The eight-step screening workflow.

    ingest -> resolve -> assemble_graph -> score_structure -> retrieve_guidance
           -> draft_memo -> verify -> human_gate -> END

The gate is the last step and it is registered as required. Nothing downstream
of it is automated: the workflow produces a recommendation, a set of citations
and an audit trail, and stops. A system that files a report or exits a customer
without a human decision is not a screening system, it is an unreviewed one.
"""

from __future__ import annotations

from typing import Any

from ..config import (
    DEFAULT_ADJUDICATION,
    DEFAULT_BLOCKING,
    DEFAULT_GRAPH,
    DEFAULT_RAG,
    DEFAULT_SCORING,
)
from ..er.adjudicate import adjudicate_all
from ..er.blocking import candidate_pairs
from ..er.resolve import accepted_pairs, cluster, cluster_summary, record_to_entity
from ..er.scoring import score_candidates
from ..graph.build import build_graph
from ..graph.features import compute_features
from ..graph.patterns import DEFAULT_THRESHOLD, assess
from ..rag.memo import build_writer, chain_view, verify_citations
from ..rag.retrieve import RegulatoryIndex
from ..registers.loaders import load_all
from .graph import END, START, StateMachine
from .spec import WorkflowError, WorkflowSpec

#: ``langgraph`` when installed, otherwise the dependency-free walker.
#: ``UBO_ENGINE`` pins it, which is what the conformance test uses to run the
#: same subject through both without rebuilding the topology.
ENGINE_ENV = "UBO_ENGINE"


def select_engine(engine: str = "auto") -> str:
    import os

    from .langgraph_engine import langgraph_available

    if engine == "auto":
        engine = os.environ.get(ENGINE_ENV, "auto")
    if engine == "auto":
        return "langgraph" if langgraph_available() else "reference"
    if engine not in {"langgraph", "reference"}:
        raise WorkflowError(f"unknown engine {engine!r}; expected 'langgraph', 'reference' or 'auto'")
    if engine == "langgraph" and not langgraph_available():
        raise WorkflowError("engine='langgraph' requested but langgraph is not installed; pip install '.[graph]'")
    return engine


def compile_workflow(spec: WorkflowSpec, engine: str = "auto", human_in_the_loop: bool = False):
    """Turn a declared topology into something with ``.invoke``."""
    resolved = select_engine(engine)
    if resolved == "langgraph":
        from .langgraph_engine import LangGraphWorkflow

        return LangGraphWorkflow(spec, human_in_the_loop=human_in_the_loop)
    if human_in_the_loop:
        raise WorkflowError("the pausing human gate needs a checkpointer; it is only on the langgraph engine")
    return StateMachine.from_spec(spec)


def build_workflow(
    regulatory: RegulatoryIndex | None = None,
    memo_backend: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    engine: str = "auto",
    human_in_the_loop: bool = False,
):
    index = regulatory if regulatory is not None else RegulatoryIndex()
    writer = build_writer(memo_backend)

    # -- 1 --------------------------------------------------------------
    def ingest(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("records") is not None:
            return {}
        records, statements = load_all()
        return {"records": records, "statements": statements}

    # -- 2 --------------------------------------------------------------
    def resolve(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("entities") is not None:
            return {}
        records = state["records"]
        by_id = {r.record_id: r for r in records}
        pairs, report = candidate_pairs(records, DEFAULT_BLOCKING)
        scores = score_candidates(by_id, pairs, DEFAULT_SCORING)
        adjudications, adj_stats = adjudicate_all(by_id, scores, cfg=DEFAULT_ADJUDICATION)
        accepted, strength = accepted_pairs(scores, adjudications)
        entities = cluster(records, accepted, strength)
        return {
            "entities": entities,
            "resolution": {
                "blocking": report.to_dict(),
                "adjudication": adj_stats,
                "clusters": cluster_summary(entities),
            },
        }

    # -- 3 --------------------------------------------------------------
    def assemble_graph(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("graph") is not None:
            return {}
        graph = build_graph(state["entities"], state["statements"], record_to_entity(state["entities"]), DEFAULT_GRAPH)
        return {"graph": graph, "graph_stats": graph.stats()}

    # -- 4 --------------------------------------------------------------
    def score_structure(state: dict[str, Any]) -> dict[str, Any]:
        graph = state["graph"]
        subject = state["subject"]
        if subject not in graph.entities:
            return {"error": f"{subject} is not an entity in the resolved graph", "features": None, "assessment": None}
        features = compute_features(graph, DEFAULT_GRAPH, roots=[subject])[subject]
        assessment = assess(features, threshold=threshold)
        return {"features": features, "assessment": assessment, "chain": chain_view(graph, subject)}

    # -- 5 --------------------------------------------------------------
    def retrieve_guidance(state: dict[str, Any]) -> dict[str, Any]:
        """Query the guidance corpus with what the structure actually shows.

        The query is assembled from the indicators that fired rather than from
        the entity name, because the applicable guidance depends on the shape of
        the finding, not on who the customer is.
        """
        entity = state["graph"].entities[state["subject"]]
        features = state.get("features")
        terms = ["beneficial ownership identification threshold"]
        if features is not None:
            if features.chain_depth >= 3:
                terms.append("indirect ownership calculated through intermediate entities")
            if features.secrecy_hops or features.secrecy_co_owners:
                terms.append("concealment techniques chains across jurisdictions")
            if features.nominee_intermediaries:
                terms.append("nominee shareholder corporate service provider on whose instructions")
            if features.in_cycle:
                terms.append("complex structures circular holdings")
            if features.sub_threshold_holdings:
                terms.append("fragmentation of holdings below the disclosure threshold")
        if entity.is_pep:
            terms.append("politically exposed person enhanced due diligence not a reason to decline")
        if entity.is_sanctioned:
            terms.append("fifty percent rule aggregate ownership blocked entity")

        sections: dict[str, dict[str, Any]] = {}
        for term in terms:
            for hit in index.search(term, DEFAULT_RAG.top_k):
                sections.setdefault(hit["id"], hit)
        return {"guidance": list(sections.values()), "guidance_queries": terms}

    # -- 6 --------------------------------------------------------------
    def draft_memo(state: dict[str, Any]) -> dict[str, Any]:
        memo = writer.write({
            "entity": state["graph"].entities[state["subject"]],
            "graph": state["graph"],
            "features": state["features"],
            "assessment": state["assessment"],
            "guidance": state.get("guidance", []),
            "chain": state.get("chain", []),
            "threshold": threshold,
        })
        return {"memo": memo}

    # -- 7 --------------------------------------------------------------
    def verify(state: dict[str, Any]) -> dict[str, Any]:
        valid_statements = {s["statement_id"] for e in state["graph"].edges for s in e.statements}
        valid_sections = set(index.by_id)
        citations = verify_citations(state["memo"], valid_statements, valid_sections)
        pep_check = state["memo"].pep_check
        return {
            "verification": {
                "citations": citations,
                "pep": pep_check,
                "passed": citations["resolution_rate"] >= 0.99 and pep_check.get("passed", True),
            }
        }

    # -- 8 --------------------------------------------------------------
    def human_gate(state: dict[str, Any]) -> dict[str, Any]:
        """The stop. Everything above it is preparation for a human decision."""
        assessment = state["assessment"]
        verification = state["verification"]
        return {
            "decision_package": {
                "subject": state["subject"],
                "recommendation": "escalate for analyst review" if assessment.flagged else "no escalation indicated",
                "structural_score": round(assessment.score, 4),
                "threshold": threshold,
                "reasons": assessment.reasons,
                "verification_passed": verification["passed"],
                "requires_human_decision": True,
                "decision": None,
                "decided_by": None,
                "decided_at": None,
            },
            "awaiting_human_decision": True,
        }

    def cannot_score(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "decision_package": {
                "subject": state["subject"],
                "recommendation": "cannot assess",
                "error": state.get("error"),
                "requires_human_decision": True,
                "decision": None,
            },
            "awaiting_human_decision": True,
        }

    # Declared once here and compiled by whichever engine is selected.
    # max_steps doubles as LangGraph's recursion_limit.
    machine = WorkflowSpec("ubo-screening", max_steps=20)
    machine.add_node("ingest", ingest)
    machine.add_node("resolve", resolve)
    machine.add_node("assemble_graph", assemble_graph)
    machine.add_node("score_structure", score_structure)
    machine.add_node("retrieve_guidance", retrieve_guidance)
    machine.add_node("draft_memo", draft_memo)
    machine.add_node("verify", verify)
    machine.add_node("human_gate", human_gate, required=True)
    machine.add_node("cannot_score", cannot_score, required=False)

    machine.add_edge(START, "ingest")
    machine.add_edge("ingest", "resolve")
    machine.add_edge("resolve", "assemble_graph")
    machine.add_edge("assemble_graph", "score_structure")
    machine.add_conditional_edges(
        "score_structure",
        lambda s: "scored" if s.get("assessment") is not None else "unscoreable",
        {"scored": "retrieve_guidance", "unscoreable": "cannot_score"},
    )
    machine.add_edge("retrieve_guidance", "draft_memo")
    machine.add_edge("draft_memo", "verify")
    machine.add_edge("verify", "human_gate")
    machine.add_edge("cannot_score", "human_gate")
    machine.add_edge("human_gate", END)
    return compile_workflow(machine, engine=engine, human_in_the_loop=human_in_the_loop)


def screen(
    subject: str,
    state: dict[str, Any] | None = None,
    regulatory: RegulatoryIndex | None = None,
    memo_backend: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    engine: str = "auto",
) -> dict[str, Any]:
    machine = build_workflow(regulatory, memo_backend, threshold, engine=engine)
    seed = dict(state or {})
    seed["subject"] = subject
    return machine.invoke(seed)


def record_decision(state: dict[str, Any], decision: str, analyst: str, timestamp: str, note: str = "") -> dict[str, Any]:
    """Close the gate with a human decision.

    Separate from the workflow on purpose. The machine cannot call this; only a
    caller holding an analyst identity can, which is what makes the gate a gate
    rather than a comment.
    """
    if decision not in {"escalate", "clear", "request_more_information"}:
        raise ValueError(f"unknown decision {decision!r}")
    package = dict(state["decision_package"])
    package.update({
        "decision": decision,
        "decided_by": analyst,
        "decided_at": timestamp,
        "note": note,
        "requires_human_decision": False,
    })
    return {**state, "decision_package": package, "awaiting_human_decision": False}
