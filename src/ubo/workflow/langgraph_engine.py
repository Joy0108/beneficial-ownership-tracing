"""The screening workflow executed on LangGraph.

This is the production engine. It compiles the shared ``WorkflowSpec`` into a
``langgraph.graph.StateGraph`` and runs it under a checkpointer, which is the
reason the library is here rather than the hand-rolled walker in ``graph.py``:

* **Reducers put the merge rule in the type.** ``_path`` and ``_checkpoints``
  are ``Annotated[list, operator.add]``, so "a node returns a partial update
  that is merged, never substituted" stops being a convention the engine
  enforces and becomes a property of the state schema. A node could not
  overwrite the decision trail even if it tried.

* **The checkpointer is the decision trail.** Every super-step is persisted, so
  an examiner asking why a subject was escalated gets the sequence of states
  that produced the decision, from the framework's own durable record rather
  than from a list this module maintains and could forget to append to. That is
  the difference between an audit trail and a summary of one.

* **Interrupts are the human gate, for real.** This workflow already had a
  ``human_gate`` node, but a node that *records* the need for a human is not
  the same as a graph that *stops*. With ``human_in_the_loop=True`` the run
  halts before the gate and ``resume()`` continues from the persisted
  checkpoint, so an analyst's decision is taken on a paused case rather than
  reconstructed after the fact. A pause that survives process death is a
  durability property, and it is not one worth hand-writing.

* **``recursion_limit`` bounds the run.** A malformed topology terminates with
  a graph error rather than spinning.

What LangGraph does *not* give us is the required-stage rule: there is no way
to declare "this run is invalid unless ``human_gate`` executed". That check
stays ours and runs after ``invoke``, against the accumulated path.
"""

from __future__ import annotations

import operator
import time
import uuid
from collections.abc import Iterator, Mapping
from typing import Annotated, Any, TypedDict

from .spec import END, START, WorkflowError, WorkflowSpec

#: State objects the checkpointer has to round-trip. Registered explicitly -
#: LangGraph blocks deserialising arbitrary classes out of a checkpoint, and it
#: is right to. A screening workflow whose state can carry anything is a
#: deserialisation gadget in a system that reads sanctions data.
ALLOWED_STATE_TYPES = [
    ("ubo.rag.memo", "Memo"),
    ("ubo.graph.build", "Edge"),
    ("ubo.graph.build", "OwnershipGraph"),
    ("ubo.graph.features", "StructuralFeatures"),
    ("ubo.graph.patterns", "RiskAssessment"),
]


class ScreeningState(TypedDict, total=False):
    """The screening state.

    Everything except the two accumulators is last-write-wins. The accumulators
    append, so the record of what ran is additive by construction.
    """

    subject: Any
    records: list[Any]
    entities: list[Any]
    entity: Any
    blocking: dict[str, Any]
    resolution: dict[str, Any]
    adjudication: dict[str, Any]
    clusters: dict[str, Any]
    statements: list[Any]
    graph: Any
    graph_stats: dict[str, Any]
    chain: list[Any]
    features: Any
    assessment: Any
    structural_score: float
    threshold: float
    pep: dict[str, Any]
    guidance: list[Any]
    guidance_queries: list[str]
    citations: list[str]
    memo: Any
    verification: dict[str, Any]
    verification_passed: bool
    passed: bool
    reasons: list[str]
    recommendation: str
    decision: str
    decision_package: dict[str, Any]
    requires_human_decision: bool
    awaiting_human_decision: bool
    error: str
    _path: Annotated[list[str], operator.add]
    _checkpoints: Annotated[list[dict[str, Any]], operator.add]


def _checkpointer() -> Any:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_STATE_TYPES))


def _assert_sentinels_match() -> None:
    from langgraph.graph import END as LG_END
    from langgraph.graph import START as LG_START

    if (START, END) != (LG_START, LG_END):
        raise WorkflowError(f"sentinel mismatch: spec uses {(START, END)}, langgraph uses {(LG_START, LG_END)}")


def _instrument(name: str, spec: WorkflowSpec):
    """Wrap a node so it records its own transition into the state."""
    fn = spec.nodes[name]

    def node(state: ScreeningState) -> dict[str, Any]:
        started = time.perf_counter()
        before = set(state)
        update = fn(dict(state)) or {}
        if not isinstance(update, Mapping):
            raise WorkflowError(f"node {name!r} returned {type(update).__name__}, expected a mapping")
        update = dict(update)
        checkpoint = {
            "step": len(state.get("_path", [])),
            "node": name,
            "next": spec.next_after(name, {**state, **update}),
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "added": sorted(set(update) - before),
        }
        return {**update, "_path": [name], "_checkpoints": [checkpoint]}

    return node


class LangGraphWorkflow:
    """A compiled LangGraph app behind the same interface as ``StateMachine``."""

    engine = "langgraph"

    def __init__(self, spec: WorkflowSpec, human_in_the_loop: bool = False):
        from langgraph.graph import StateGraph

        _assert_sentinels_match()
        spec.validate()
        self.spec = spec
        self.name = spec.name
        self.max_steps = spec.max_steps
        self.human_in_the_loop = human_in_the_loop

        builder = StateGraph(ScreeningState)
        for node_name in spec.nodes:
            builder.add_node(node_name, _instrument(node_name, spec))
        builder.add_edge(START, spec.entry)
        for src, dst in spec.edges.items():
            builder.add_edge(src, dst)
        for src, (router, mapping) in spec.conditional.items():
            builder.add_conditional_edges(src, router, mapping)

        # Stop *before* the gate, so the analyst decides on a paused case with
        # the memo and its verification in front of them, rather than reviewing
        # a decision the workflow already recorded.
        interrupt_before = ["human_gate"] if human_in_the_loop and "human_gate" in spec.nodes else []
        self.app = builder.compile(checkpointer=_checkpointer(), interrupt_before=interrupt_before)

    # -- execution ---------------------------------------------------------
    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": self.spec.max_steps}

    def _finish(self, result: Mapping[str, Any], thread_id: str, *, enforce: bool) -> dict[str, Any]:
        state = dict(result)
        state.setdefault("_path", [])
        state.setdefault("_checkpoints", [])
        state["_thread_id"] = thread_id
        state["_engine"] = self.engine
        state["_steps"] = len(state["_path"])
        if enforce:
            missing = self.spec.missing_required(state["_path"])
            if missing:
                raise WorkflowError(f"required stage(s) did not run: {missing}; path={state['_path']}")
        return state

    def invoke(self, state: Mapping[str, Any], thread_id: str | None = None) -> dict[str, Any]:
        thread_id = thread_id or self._thread_id_for(state)
        try:
            result = self.app.invoke(dict(state), config=self._config(thread_id))
        except Exception as exc:
            if type(exc).__name__ == "GraphRecursionError":
                raise WorkflowError(
                    f"{self.name} exceeded max_steps={self.spec.max_steps}"
                ) from exc
            raise
        # An interrupted run has not finished, so the required-stage rule does
        # not apply to it yet. It applies to what `resume` returns.
        return self._finish(result, thread_id, enforce=not self.interrupted(thread_id))

    def resume(self, thread_id: str, update: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Continue a run paused at the human gate, optionally recording a decision."""
        config = self._config(thread_id)
        if update:
            self.app.update_state(config, dict(update))
        return self._finish(self.app.invoke(None, config=config), thread_id, enforce=True)

    def interrupted(self, thread_id: str) -> bool:
        return bool(self.app.get_state(self._config(thread_id)).next)

    def stream(self, state: Mapping[str, Any], thread_id: str | None = None) -> Iterator[dict[str, Any]]:
        thread_id = thread_id or self._thread_id_for(state)
        yield from self.app.stream(dict(state), config=self._config(thread_id), stream_mode="updates")

    # -- introspection -----------------------------------------------------
    def state_history(self, thread_id: str) -> list[dict[str, Any]]:
        """The framework's own checkpoint record, oldest first."""
        entries = []
        for snapshot in reversed(list(self.app.get_state_history(self._config(thread_id)))):
            path = list(snapshot.values.get("_path", []))
            entries.append({
                "step": snapshot.metadata.get("step") if snapshot.metadata else None,
                "completed": path[-1] if path else None,
                "next": list(snapshot.next),
                "path": path,
            })
        return entries

    def to_mermaid(self) -> str:
        return self.app.get_graph().draw_mermaid()

    def validate(self) -> None:
        self.spec.validate()

    @staticmethod
    def _thread_id_for(state: Mapping[str, Any]) -> str:
        subject = state.get("subject")
        if isinstance(subject, Mapping):
            base = subject.get("name") or subject.get("id") or "subject"
        else:
            base = str(subject or "subject")
        return f"{base}:{uuid.uuid4().hex[:8]}"


def langgraph_available() -> bool:
    try:
        import langgraph.graph  # noqa: F401
    except Exception:
        return False
    return True


__all__ = ["ALLOWED_STATE_TYPES", "LangGraphWorkflow", "ScreeningState", "langgraph_available"]
