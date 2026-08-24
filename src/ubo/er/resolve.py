"""Clustering and the resolved entity spine.

Matched pairs become clusters by transitive closure, which is where a single
false merge does its damage: one bad edge joins two structures and every
ownership chain through them becomes wrong. The closure is therefore computed
with the weakest link recorded, so a cluster can be inspected by the strength of
the edge that is actually holding it together.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..registers.loaders import Record
from .adjudicate import Adjudication
from .scoring import PairScore


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {i: i for i in items}
        self.rank = dict.fromkeys(self.parent, 0)

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


@dataclass
class ResolvedEntity:
    entity_id: str
    entity_type: str
    canonical_name: str
    record_ids: list[str]
    sources: list[str]
    jurisdictions: list[str]
    is_sanctioned: bool = False
    is_pep: bool = False
    pep_position: str = ""
    weakest_link: float = 1.0
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "canonical_name": self.canonical_name,
            "record_ids": self.record_ids,
            "sources": self.sources,
            "jurisdictions": self.jurisdictions,
            "is_sanctioned": self.is_sanctioned,
            "is_pep": self.is_pep,
            "pep_position": self.pep_position,
            "weakest_link": round(self.weakest_link, 4),
            "provenance": self.provenance,
        }


def accepted_pairs(
    scores: Sequence[PairScore], adjudications: Sequence[Adjudication] = ()
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], float]]:
    """Pairs to merge, with the strength that carried each one."""
    verdicts = {(a.left, a.right): a for a in adjudications}
    pairs: set[tuple[str, str]] = set()
    strength: dict[tuple[str, str], float] = {}

    for score in scores:
        key = (score.left, score.right)
        if score.decision == "match":
            pairs.add(key)
            strength[key] = score.score
        elif score.decision == "review":
            verdict = verdicts.get(key)
            if verdict is not None and verdict.decision == "match":
                pairs.add(key)
                strength[key] = min(score.score, verdict.confidence)
    return pairs, strength


def cluster(
    records: Sequence[Record],
    pairs: set[tuple[str, str]],
    strength: dict[tuple[str, str], float] | None = None,
) -> list[ResolvedEntity]:
    strength = strength or {}
    by_id = {r.record_id: r for r in records}
    uf = UnionFind(by_id)

    weakest: dict[str, float] = {}
    for a, b in sorted(pairs):
        if a not in by_id or b not in by_id:
            continue
        uf.union(a, b)
        s = strength.get((a, b), 1.0)
        root = uf.find(a)
        weakest[root] = min(weakest.get(root, 1.0), s)

    groups: dict[str, list[str]] = defaultdict(list)
    for record_id in by_id:
        groups[uf.find(record_id)].append(record_id)

    entities: list[ResolvedEntity] = []
    for root, members in sorted(groups.items()):
        members.sort()
        rows = [by_id[m] for m in members]
        entities.append(
            ResolvedEntity(
                entity_id=f"E-{root}",
                entity_type=rows[0].entity_type,
                canonical_name=_canonical_name(rows),
                record_ids=members,
                sources=sorted({r.source for r in rows}),
                jurisdictions=sorted({r.jurisdiction for r in rows if r.jurisdiction}),
                is_sanctioned=any(r.is_sanctioned for r in rows),
                is_pep=any(r.is_pep for r in rows),
                pep_position=next((r.raw.get("position", "") for r in rows if r.raw.get("position")), ""),
                weakest_link=min(weakest.get(uf.find(root), 1.0), 1.0),
                provenance=[{"record_id": r.record_id, "source": r.source, "name_as_recorded": r.name} for r in rows],
            )
        )
    return entities


def _canonical_name(rows: Sequence[Record]) -> str:
    """Prefer the register that publishes legal names, then the longest form.

    GLEIF and Companies House publish the registered legal name. Aggregator
    extracts and sanctions lists publish whatever they scraped, which is where
    the typos are.
    """
    priority = {"gleif_l1": 0, "companies_house": 1, "opensanctions": 2, "openownership": 3, "ofac": 4, "aggregator": 5}
    best = sorted(rows, key=lambda r: (priority.get(r.source, 9), -len(r.name)))
    return best[0].name


def record_to_entity(entities: Sequence[ResolvedEntity]) -> dict[str, str]:
    return {rid: e.entity_id for e in entities for rid in e.record_ids}


def cluster_summary(entities: Sequence[ResolvedEntity]) -> dict[str, Any]:
    sizes = [len(e.record_ids) for e in entities]
    multi = [e for e in entities if len(e.record_ids) > 1]
    return {
        "entities": len(entities),
        "records": sum(sizes),
        "singletons": sum(1 for s in sizes if s == 1),
        "merged_entities": len(multi),
        "largest_cluster": max(sizes) if sizes else 0,
        "cross_source_entities": sum(1 for e in entities if len(e.sources) > 1),
        "sanctioned_entities": sum(1 for e in entities if e.is_sanctioned),
        "pep_entities": sum(1 for e in entities if e.is_pep),
        "weakest_link_min": round(min([e.weakest_link for e in multi], default=1.0), 4),
    }
