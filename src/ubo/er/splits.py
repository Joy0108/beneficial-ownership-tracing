"""Train/test splitting, and the leakage that makes record-linkage numbers lie.

The standard way to evaluate a matcher is to split the labelled *pairs*. It is
also wrong, and wrong in the direction that flatters you.

Consider one real entity with four register records: A, B, C, D. That is six
positive pairs. Split them at random and AB, AC land in train while BD, CD land
in test. The test pairs are not new entities - they are the same four records,
whose names, addresses and identifiers the model has already been fitted to. Any
threshold or weight tuned on the train half transfers to the test half for free,
and the reported F1 measures memorisation of specific strings rather than the
ability to resolve an entity the model has never seen.

Splitting by *cluster* fixes it: an entity is wholly in train or wholly in test,
so a test pair involves records the model has never touched. The two numbers are
computed side by side in :func:`leakage_report` because the gap between them is
the size of the illusion, and it is worth knowing.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .blocking import true_pairs


@dataclass
class Split:
    name: str
    train_pairs: set[tuple[str, str]]
    test_pairs: set[tuple[str, str]]
    train_records: set[str]
    test_records: set[str]

    @property
    def shared_records(self) -> set[str]:
        return self.train_records & self.test_records

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.name,
            "train_pairs": len(self.train_pairs),
            "test_pairs": len(self.test_pairs),
            "train_records": len(self.train_records),
            "test_records": len(self.test_records),
            "records_in_both_halves": len(self.shared_records),
            "leaks": bool(self.shared_records),
        }


def pair_level_split(
    clusters: dict[str, Iterable[str]], test_fraction: float = 0.4, seed: int = 7
) -> Split:
    """The naive split. Kept so the leak can be measured, not so it can be used."""
    positives = sorted(true_pairs(clusters))
    rng = random.Random(seed)
    rng.shuffle(positives)
    cut = int(len(positives) * (1 - test_fraction))
    train, test = set(positives[:cut]), set(positives[cut:])
    return Split(
        "pair_level",
        train, test,
        {r for pair in train for r in pair},
        {r for pair in test for r in pair},
    )


def cluster_level_split(
    clusters: dict[str, Iterable[str]], test_fraction: float = 0.4, seed: int = 7
) -> Split:
    """Whole entities go to one side or the other. No record appears in both."""
    ids = sorted(clusters)
    rng = random.Random(seed)
    rng.shuffle(ids)
    cut = int(len(ids) * (1 - test_fraction))
    train_ids, test_ids = set(ids[:cut]), set(ids[cut:])

    def pairs_of(subset: set[str]) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for cid in subset:
            members = sorted(set(clusters[cid]))
            out.update(combinations(members, 2))
        return out

    return Split(
        "cluster_level",
        pairs_of(train_ids), pairs_of(test_ids),
        {r for cid in train_ids for r in clusters[cid]},
        {r for cid in test_ids for r in clusters[cid]},
    )


def negatives_for(
    split: Split, candidates: set[tuple[str, str]], positives: set[tuple[str, str]], half: str = "test"
) -> set[tuple[str, str]]:
    """Candidate pairs in one half of the split that are not true matches.

    Negatives are drawn from the *candidate* set rather than from all pairs. A
    matcher is only ever asked about pairs blocking proposed, so scoring it on
    pairs it would never see inflates precision with free rejections.
    """
    records = split.test_records if half == "test" else split.train_records
    return {(a, b) for a, b in candidates if a in records and b in records and (a, b) not in positives}


def evaluate_on(
    pairs_predicted: set[tuple[str, str]], positives: set[tuple[str, str]], negatives: set[tuple[str, str]]
) -> dict[str, float]:
    tp = len(pairs_predicted & positives)
    fp = len(pairs_predicted & negatives)
    fn = len(positives - pairs_predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def leakage_report(
    clusters: dict[str, Iterable[str]],
    predicted: set[tuple[str, str]],
    candidates: set[tuple[str, str]],
    test_fraction: float = 0.4,
    seed: int = 7,
) -> dict[str, Any]:
    """Score the same predictions under both splits and report the gap."""
    positives = true_pairs(clusters)
    out: dict[str, Any] = {}
    for split in (pair_level_split(clusters, test_fraction, seed), cluster_level_split(clusters, test_fraction, seed)):
        test_positives = split.test_pairs
        test_negatives = negatives_for(split, candidates, positives, "test")
        predicted_in_half = {
            p for p in predicted if p[0] in split.test_records and p[1] in split.test_records
        }
        out[split.name] = {
            **split.to_dict(),
            **evaluate_on(predicted_in_half, test_positives, test_negatives),
        }

    pair_f1 = out["pair_level"]["f1"]
    cluster_f1 = out["cluster_level"]["f1"]
    out["inflation"] = {
        "pair_level_f1": pair_f1,
        "cluster_level_f1": cluster_f1,
        "absolute_gap": round(pair_f1 - cluster_f1, 4),
        "records_leaked_by_pair_split": out["pair_level"]["records_in_both_halves"],
        "note": (
            "The pair-level split shares records between train and test; the cluster-level split does not. "
            "The gap is how much a pair-level benchmark would have overstated this matcher."
        ),
    }
    return out


def stratify_by_type(clusters: dict[str, Sequence[str]], record_type: dict[str, str]) -> dict[str, dict[str, list[str]]]:
    """Split the clustering into person and company sub-problems.

    They fail differently - people collide on common names, companies collide on
    trading names across jurisdictions - so a combined F1 hides which one broke.
    """
    out: dict[str, dict[str, list[str]]] = {"person": {}, "company": {}}
    for cid, members in clusters.items():
        members = list(members)
        kind = record_type.get(members[0], "company")
        out.setdefault(kind, {})[cid] = members
    return out
