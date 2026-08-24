"""Ground truth from the seed world, and the mapping back onto resolved entities.

The seed world labels *seed* entities (P001, C004). The pipeline produces
*resolved* entities keyed by whichever record happened to become the union-find
root. Evaluating the graph stage therefore needs a bridge between the two, and
building it carelessly is a way to grade a pipeline on its own output.

The bridge here is majority vote over the records in each resolved cluster, and
it explicitly reports impurity: a resolved entity whose records come from two
different seed entities is a false merge, and it is counted rather than resolved
away. Nothing outside this module reads the seed world.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import WORLD_PATH
from ..er.resolve import ResolvedEntity


def load_world(path: Path = WORLD_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def truth_clusters(world: dict[str, Any], multi_only: bool = True) -> dict[str, list[str]]:
    clusters = world["truth_clusters"]
    return {k: v for k, v in clusters.items() if not multi_only or len(v) > 1}


def record_to_seed(world: dict[str, Any]) -> dict[str, str]:
    return {rid: seed for seed, rids in world["truth_clusters"].items() for rid in rids}


def map_entities_to_seed(
    entities: Sequence[ResolvedEntity], world: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Resolved entity id -> seed entity id, by majority vote over its records."""
    lookup = record_to_seed(world)
    mapping: dict[str, str] = {}
    impure = []
    for entity in entities:
        seeds = Counter(lookup[r] for r in entity.record_ids if r in lookup)
        if not seeds:
            continue
        winner, count = seeds.most_common(1)[0]
        mapping[entity.entity_id] = winner
        if len(seeds) > 1:
            impure.append({
                "entity_id": entity.entity_id,
                "canonical_name": entity.canonical_name,
                "seed_entities_merged": dict(seeds),
                "purity": round(count / sum(seeds.values()), 3),
            })

    # A seed entity split across several resolved entities is the other failure
    # mode, and it is the one that breaks ownership chains rather than fusing
    # unrelated ones.
    fragments = Counter(mapping.values())
    return mapping, {
        "resolved_entities_mapped": len(mapping),
        "false_merges": len(impure),
        "false_merge_detail": impure[:10],
        "fragmented_seed_entities": sum(1 for c in fragments.values() if c > 1),
        "worst_fragmentation": max(fragments.values(), default=0),
    }


def seed_to_entity(mapping: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for entity_id, seed in mapping.items():
        out.setdefault(seed, []).append(entity_id)
    return out


def layering_labels(
    world: dict[str, Any], mapping: dict[str, str]
) -> dict[str, bool]:
    """Labels lifted onto resolved entity ids.

    A labelled seed entity that fragmented into several resolved entities gets
    every fragment labelled. That is deliberately unkind to the model: if entity
    resolution split a structure, the graph stage has to find the risk in a
    partial chain, and pretending otherwise would hide an upstream failure
    inside a downstream score.
    """
    labels = world.get("layered_roots", {})
    reverse = seed_to_entity(mapping)
    out: dict[str, bool] = {}
    for seed, value in labels.items():
        for entity_id in reverse.get(seed, []):
            out[entity_id] = bool(value)
    return out


def structural_entities(world: dict[str, Any]) -> set[str]:
    return set(world.get("structural_entities", []))
