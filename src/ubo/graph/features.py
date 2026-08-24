"""Structural risk features.

Each feature is one thing a financial-crime analyst would look at, computed so
that the number can be traced back to specific edges. None of them is a verdict:
a long chain through Cyprus is how a great many legitimate European groups are
built. What the model is asked to find is the *combination* that has no
commercial explanation - depth plus secrecy hops plus a nominee intermediary
plus circularity - and the features are kept separate so that a decision can say
which of them fired.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ..config import DEFAULT_GRAPH, SECRECY_JURISDICTIONS, GraphConfig
from .build import OwnershipGraph, effective_ownership

FEATURE_NAMES = (
    "chain_depth",
    "controlled_count",
    "jurisdiction_hops",
    "secrecy_hops",
    "secrecy_ratio",
    "in_cycle",
    "max_intermediary_centrality",
    "shared_intermediaries",
    "nominee_intermediaries",
    "secrecy_co_owners",
    "mean_edge_confidence",
    "ownership_attenuation",
    "sub_threshold_holdings",
    "single_source_edges",
)


@dataclass
class StructuralFeatures:
    entity_id: str
    chain_depth: float = 0.0
    controlled_count: float = 0.0
    jurisdiction_hops: float = 0.0
    secrecy_hops: float = 0.0
    secrecy_ratio: float = 0.0
    in_cycle: float = 0.0
    max_intermediary_centrality: float = 0.0
    shared_intermediaries: float = 0.0
    nominee_intermediaries: float = 0.0
    secrecy_co_owners: float = 0.0
    mean_edge_confidence: float = 1.0
    ownership_attenuation: float = 0.0
    sub_threshold_holdings: float = 0.0
    single_source_edges: float = 0.0

    def vector(self) -> list[float]:
        return [getattr(self, name) for name in FEATURE_NAMES]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def betweenness(graph: OwnershipGraph, max_depth: int = 8) -> dict[str, float]:
    """How often an entity sits on someone else's control path.

    A corporate service provider or a nominee director shows up here as a node
    that lies between many unrelated owners and their assets. It is the feature
    that finds the shared intermediary - the thing that links two structures
    that otherwise have nothing in common.
    """
    counts: dict[str, float] = defaultdict(float)
    total_paths = 0
    for root in graph.roots():
        for path in graph.paths(root, max_depth):
            total_paths += 1
            for edge in path[:-1]:  # endpoints are not intermediaries
                counts[edge.child] += 1.0
    if not total_paths:
        return {}
    return {node: round(count / total_paths, 6) for node, count in counts.items()}


def _cycle_members(graph: OwnershipGraph) -> set[str]:
    return {node for cycle in graph.cycles() for node in cycle}


NOMINEE_NAME_MARKERS = ("nominee", "trust", "fiduciar", "secretaria", "corporate services", "management services")


def roots_served(graph: OwnershipGraph, max_depth: int = 8) -> dict[str, set[str]]:
    """For each node, the set of distinct roots whose control paths cross it."""
    out: dict[str, set[str]] = defaultdict(set)
    for root in graph.roots():
        for path in graph.paths(root, max_depth):
            for edge in path[:-1]:
                out[edge.child].add(root)
    return out


def compute_features(
    graph: OwnershipGraph, cfg: GraphConfig = DEFAULT_GRAPH, roots: Sequence[str] | None = None
) -> dict[str, StructuralFeatures]:
    centrality = betweenness(graph)
    in_cycle = _cycle_members(graph)
    served = roots_served(graph, cfg.max_chain_depth)
    targets = list(roots) if roots is not None else graph.roots()

    out: dict[str, StructuralFeatures] = {}
    for root in targets:
        paths = graph.paths(root, cfg.max_chain_depth)
        descendants = graph.descendants(root, cfg.max_chain_depth)
        feats = StructuralFeatures(entity_id=root)

        if not paths:
            out[root] = feats
            continue

        feats.chain_depth = float(max(len(p) for p in paths))
        feats.controlled_count = float(len(descendants))

        # Jurisdiction and secrecy hops are counted on the deepest chain, which
        # is the one a structuring exercise actually builds.
        deepest = max(paths, key=len)
        chain_nodes = [root] + [e.child for e in deepest]
        jurisdictions = [graph.jurisdiction_of(n) for n in chain_nodes]
        feats.jurisdiction_hops = float(
            sum(1 for a, b in zip(jurisdictions, jurisdictions[1:], strict=False) if a and b and a != b)
        )
        secrecy_flags = [j in SECRECY_JURISDICTIONS for j in jurisdictions]
        feats.secrecy_hops = float(sum(secrecy_flags))
        feats.secrecy_ratio = round(sum(secrecy_flags) / len(chain_nodes), 4)

        feats.in_cycle = 1.0 if any(n in in_cycle for n in chain_nodes) else 0.0
        feats.max_intermediary_centrality = max((centrality.get(n, 0.0) for n in chain_nodes[1:-1]), default=0.0)

        # Two separate things, because conflating them produces false flags on
        # perfectly ordinary structures.
        #
        # A *shared* intermediary sits on the control paths of more than one
        # root. That covers legitimate co-ownership: a joint venture with two
        # shareholders is shared and means nothing on its own.
        #
        # A *nominee* is a shared intermediary that is also doing something a
        # nominee does - sitting in a secrecy jurisdiction, or trading under a
        # name that advertises the service. That is the signal worth weighting.
        middle = chain_nodes[1:-1]
        feats.shared_intermediaries = float(sum(1 for n in middle if len(served.get(n, ())) > 1))
        feats.nominee_intermediaries = float(
            sum(
                1
                for n in middle
                if _looks_like_nominee(graph, n) and len(served.get(n, ())) > 1
            )
        )

        # Who else holds the companies on this chain. A holding co-mingled with
        # an offshore co-owner is opaque even when the chain itself is short and
        # entirely onshore - the opacity is upstream, in the counterparty, not in
        # the path. Without this the chain-shape features cannot see it at all.
        feats.secrecy_co_owners = float(
            len({
                edge.parent
                for node in chain_nodes[1:]
                for edge in graph.parents(node, control_only=True)
                if edge.parent != root and edge.parent not in chain_nodes and graph.is_secrecy(edge.parent)
            })
        )

        edges = [e for p in paths for e in p]
        feats.mean_edge_confidence = round(sum(e.confidence for e in edges) / len(edges), 4)
        feats.single_source_edges = float(sum(1 for e in edges if len(e.statements) == 1)) / len(edges)

        effective = effective_ownership(graph, root, cfg)
        if effective:
            direct = max((e.share_percent for e in graph.children(root, control_only=True)), default=0.0)
            deepest_effective = min(effective.values())
            feats.ownership_attenuation = round(max(0.0, direct - deepest_effective), 4)
            # Holdings that sit just under the disclosure floor are the shape a
            # structure takes when it is built to avoid being disclosed.
            feats.sub_threshold_holdings = float(
                sum(1 for v in effective.values() if 0 < v < cfg.control_threshold)
            )

        out[root] = feats
    return out


def _looks_like_nominee(graph: OwnershipGraph, node: str) -> bool:
    entity = graph.entities.get(node)
    if entity is None:
        return False
    name = entity.canonical_name.lower()
    if any(marker in name for marker in NOMINEE_NAME_MARKERS):
        return True
    return graph.is_secrecy(node)


def feature_matrix(features: dict[str, StructuralFeatures]) -> tuple[list[str], list[list[float]]]:
    ids = sorted(features)
    return ids, [features[i].vector() for i in ids]


def describe(features: StructuralFeatures) -> list[str]:
    """Plain-language reasons, for the memo. Only what actually fired."""
    lines = []
    if features.chain_depth >= 4:
        lines.append(f"control passes through {int(features.chain_depth)} layers before reaching the operating company")
    if features.secrecy_hops >= 2:
        lines.append(f"{int(features.secrecy_hops)} entities on the deepest chain sit in secrecy jurisdictions")
    if features.jurisdiction_hops >= 3:
        lines.append(f"the chain crosses {int(features.jurisdiction_hops)} jurisdictions")
    if features.in_cycle:
        lines.append("the structure contains circular ownership, so no single filing shows the whole picture")
    if features.nominee_intermediaries:
        lines.append(
            f"{int(features.nominee_intermediaries)} intermediaries on this chain act for more than one owner "
            "and are either registered in a secrecy jurisdiction or trade as a nominee or trust service")
    if features.secrecy_co_owners:
        lines.append(
            f"{int(features.secrecy_co_owners)} companies on this chain are co-owned by entities registered "
            "in secrecy jurisdictions that are outside the chain itself")
    if features.max_intermediary_centrality >= 0.1:
        lines.append("an intermediary on this chain also sits on a large share of unrelated control paths")
    if features.sub_threshold_holdings:
        lines.append(
            f"{int(features.sub_threshold_holdings)} effective holdings fall below the disclosure threshold "
            "once attenuation along the chain is applied")
    if features.mean_edge_confidence < 0.8:
        lines.append(f"the chain rests on edges with mean confidence {features.mean_edge_confidence}")
    return lines
