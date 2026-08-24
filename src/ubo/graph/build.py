"""The ownership and control graph, with provenance on every edge.

Nodes are resolved entities, not register records: that is the whole point of
the resolution stage. Edges are ownership, control, directorship or accounting
consolidation, and each one keeps the list of statements that assert it - which
register, which statement id, when it was retrieved, and how confident the
resolution of both endpoints was.

That last part matters more than it looks. An ownership edge between two
entities is only as trustworthy as the weakest merge that produced its
endpoints, so edge confidence carries the resolution confidence forward instead
of quietly discarding it. A chain assembled from four edges each resting on a
0.6-confidence merge is not a 100% chain, and a memo that says so is worth more
than one that does not.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import DEFAULT_GRAPH, SECRECY_JURISDICTIONS, GraphConfig
from ..er.resolve import ResolvedEntity
from ..registers.loaders import Statement

CONTROL_KINDS = {"shareholding", "ownership", "consolidation"}


@dataclass
class Edge:
    """Interested party -> subject. The arrow points the way control flows."""

    parent: str
    child: str
    interest_type: str
    share_percent: float
    confidence: float
    statements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_control(self) -> bool:
        return self.interest_type in CONTROL_KINDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent": self.parent,
            "child": self.child,
            "interest_type": self.interest_type,
            "share_percent": self.share_percent,
            "confidence": round(self.confidence, 4),
            "statements": self.statements,
        }


@dataclass
class OwnershipGraph:
    entities: dict[str, ResolvedEntity]
    edges: list[Edge]
    unresolved_statements: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.out: dict[str, list[Edge]] = defaultdict(list)
        self.into: dict[str, list[Edge]] = defaultdict(list)
        for edge in self.edges:
            self.out[edge.parent].append(edge)
            self.into[edge.child].append(edge)

    # -- traversal ---------------------------------------------------------
    def children(self, entity_id: str, control_only: bool = False) -> list[Edge]:
        return [e for e in self.out.get(entity_id, []) if not control_only or e.is_control]

    def parents(self, entity_id: str, control_only: bool = False) -> list[Edge]:
        return [e for e in self.into.get(entity_id, []) if not control_only or e.is_control]

    def roots(self) -> list[str]:
        """Entities nobody owns. The candidate ultimate beneficial owners."""
        return sorted(e for e in self.entities if not self.into.get(e))

    def descendants(self, entity_id: str, max_depth: int = 12) -> dict[str, int]:
        """Reachable entities and the depth at which each is first reached.

        Breadth-first, so the recorded depth is the shortest control path. A
        cycle is visited once, which is what stops circular ownership - a real
        and deliberate structure - from making traversal non-terminating.
        """
        seen = {entity_id: 0}
        queue = deque([(entity_id, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self.children(node):
                if edge.child not in seen:
                    seen[edge.child] = depth + 1
                    queue.append((edge.child, depth + 1))
        seen.pop(entity_id, None)
        return seen

    def paths(self, entity_id: str, max_depth: int = 12, limit: int = 200) -> list[list[Edge]]:
        """Every simple control path from an entity downward."""
        out: list[list[Edge]] = []

        def walk(node: str, trail: list[Edge], visited: set[str]) -> None:
            if len(out) >= limit or len(trail) >= max_depth:
                return
            for edge in self.children(node, control_only=True):
                if edge.child in visited:
                    continue
                path = trail + [edge]
                out.append(path)
                walk(edge.child, path, visited | {edge.child})

        walk(entity_id, [], {entity_id})
        return out

    def cycles(self, max_length: int = 8) -> list[list[str]]:
        """Circular ownership, reported once per cycle.

        Circularity is legitimate in some group structures and a classic
        obscuring device in others, so it is detected and reported rather than
        scored as guilt on its own.
        """
        found: dict[frozenset[str], list[str]] = {}

        def walk(start: str, node: str, trail: list[str], visited: set[str]) -> None:
            if len(trail) > max_length:
                return
            for edge in self.children(node, control_only=True):
                if edge.child == start and len(trail) >= 2:
                    key = frozenset(trail)
                    found.setdefault(key, trail + [start])
                elif edge.child not in visited:
                    walk(start, edge.child, trail + [edge.child], visited | {edge.child})

        for entity in sorted(self.entities):
            walk(entity, entity, [entity], {entity})
        return sorted(found.values(), key=len)

    def jurisdiction_of(self, entity_id: str) -> str:
        entity = self.entities.get(entity_id)
        if entity is None or not entity.jurisdictions:
            return ""
        return entity.jurisdictions[0]

    def is_secrecy(self, entity_id: str) -> bool:
        return self.jurisdiction_of(entity_id) in SECRECY_JURISDICTIONS

    def stats(self) -> dict[str, Any]:
        control = [e for e in self.edges if e.is_control]
        return {
            "nodes": len(self.entities),
            "edges": len(self.edges),
            "control_edges": len(control),
            "roots": len(self.roots()),
            "cycles": len(self.cycles()),
            "unresolved_statements": len(self.unresolved_statements),
            "mean_edge_confidence": round(sum(e.confidence for e in self.edges) / len(self.edges), 4) if self.edges else 0.0,
            "secrecy_nodes": sum(1 for e in self.entities if self.is_secrecy(e)),
        }


def build_graph(
    entities: Sequence[ResolvedEntity],
    statements: Iterable[Statement],
    record_to_entity: dict[str, str],
    cfg: GraphConfig = DEFAULT_GRAPH,
) -> OwnershipGraph:
    by_id = {e.entity_id: e for e in entities}
    merged: dict[tuple[str, str, str], Edge] = {}
    unresolved: list[dict[str, Any]] = []

    for st in statements:
        parent = record_to_entity.get(st.interested_record_id)
        child = record_to_entity.get(st.subject_record_id)
        if parent is None or child is None:
            # A statement whose endpoints did not resolve is dropped from the
            # graph and kept in a list. Silently discarding it would make the
            # graph look complete when it is not.
            unresolved.append({**st.to_dict(), "reason": "endpoint did not resolve to an entity"})
            continue
        if parent == child:
            unresolved.append({**st.to_dict(), "reason": "self-loop after resolution, likely a false merge"})
            continue

        # An edge is only as good as the merges holding its endpoints.
        endpoint_confidence = min(
            by_id[parent].weakest_link if parent in by_id else 1.0,
            by_id[child].weakest_link if child in by_id else 1.0,
        )
        key = (parent, child, st.interest_type)
        edge = merged.get(key)
        if edge is None:
            merged[key] = Edge(
                parent=parent, child=child, interest_type=st.interest_type,
                share_percent=st.share_percent,
                confidence=round(st.confidence * endpoint_confidence, 6),
                statements=[st.to_dict()],
            )
        else:
            # Two registers asserting the same edge is corroboration, so the
            # edge gets the better confidence and keeps both statements.
            edge.statements.append(st.to_dict())
            edge.confidence = max(edge.confidence, st.confidence * endpoint_confidence)
            edge.share_percent = max(edge.share_percent, st.share_percent)

    return OwnershipGraph(entities=by_id, edges=list(merged.values()), unresolved_statements=unresolved)


def effective_ownership(graph: OwnershipGraph, root: str, cfg: GraphConfig = DEFAULT_GRAPH) -> dict[str, float]:
    """Percentage held through every path, attenuated along each chain.

    60% of a company that holds 51% of another is 30.6%, not 51%. Disclosure
    regimes are written in terms of effective holding, and reporting the direct
    percentage at depth is how a structure designed to sit under a threshold
    gets recorded as sitting over it.
    """
    totals: dict[str, float] = defaultdict(float)
    for path in graph.paths(root, cfg.max_chain_depth):
        share = 1.0
        for edge in path:
            pct = edge.share_percent / 100.0 if edge.share_percent else 0.0
            share = share * pct if cfg.attenuate_ownership else max(share, pct)
        if share > 0:
            totals[path[-1].child] = max(totals[path[-1].child], share * 100.0)
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))
