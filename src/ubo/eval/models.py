"""Comparing the engineered features against the graph neural network.

Trained and scored on the same eleven labelled structures the GCN reports a
perfect F1. It has 128 parameters and 11 labels; that number measures nothing
except that the model can memorise its training set, and reporting it would be
straightforwardly dishonest.

Leave-one-out is the only defensible protocol at this size: hold out one
structure, train on the rest, predict the held-out one, repeat. It is expensive
and it is noisy - with eleven folds a single flip moves F1 by roughly 0.09 - but
it measures generalisation rather than recall of the training set, and the
confidence interval it implies is reported alongside the point estimate so the
noise is visible rather than hidden.
"""

from __future__ import annotations

import math
from typing import Any

from ..graph.build import OwnershipGraph
from ..graph.features import StructuralFeatures
from ..graph.gnn import train_gcn
from ..graph.patterns import DEFAULT_THRESHOLD, RULE_WEIGHTS, assess


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval. At n=11 the normal approximation is not usable."""
    if not total:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


def rules_in_sample(
    features: dict[str, StructuralFeatures], labels: dict[str, bool], threshold: float = DEFAULT_THRESHOLD
) -> dict[str, Any]:
    tp = fp = fn = 0
    for eid, truth in labels.items():
        if eid not in features:
            continue
        flagged = assess(features[eid], RULE_WEIGHTS, threshold).flagged
        tp += flagged and truth
        fp += flagged and not truth
        fn += (not flagged) and truth
    return {"protocol": "in-sample", "note": "weights are hand-set, not fitted, so this is not a training score",
            **_prf(tp, fp, fn)}


def gcn_in_sample(graph: OwnershipGraph, labels: dict[str, bool], **kwargs) -> dict[str, Any]:
    result = train_gcn(graph, labels, **kwargs)
    tp = fp = fn = 0
    for eid, truth in labels.items():
        flagged = result.scores.get(eid, 0.0) >= 0.5
        tp += flagged and truth
        fp += flagged and not truth
        fn += (not flagged) and truth
    return {
        "protocol": "in-sample (trained on these same labels)",
        "warning": "meaningless as a quality measure; kept only to show the gap against leave-one-out",
        **_prf(tp, fp, fn),
        **result.to_dict(),
    }


def gcn_leave_one_out(graph: OwnershipGraph, labels: dict[str, bool], **kwargs) -> dict[str, Any]:
    tp = fp = fn = 0
    correct = 0
    per_fold = []
    for held_out in sorted(labels):
        train_labels = {k: v for k, v in labels.items() if k != held_out}
        result = train_gcn(graph, train_labels, **kwargs)
        score = result.scores.get(held_out, 0.0)
        flagged = score >= 0.5
        truth = labels[held_out]
        tp += flagged and truth
        fp += flagged and not truth
        fn += (not flagged) and truth
        correct += flagged == truth
        per_fold.append({"held_out": held_out, "score": round(score, 4), "predicted": flagged, "actual": truth})
    lo, hi = _wilson(correct, len(labels))
    return {
        "protocol": "leave-one-out",
        "folds": len(labels),
        "accuracy": round(correct / len(labels), 4) if labels else 0.0,
        "accuracy_95ci": [lo, hi],
        **_prf(tp, fp, fn),
        "per_fold": per_fold,
    }


def rules_leave_one_out(
    features: dict[str, StructuralFeatures], labels: dict[str, bool], threshold: float = DEFAULT_THRESHOLD
) -> dict[str, Any]:
    """For symmetry.

    The rule weights do not depend on the labels at all, so leaving one out
    changes nothing and this is identical to the in-sample number. That is not a
    trick - it is the actual advantage of a model with no fitted parameters, and
    it is why the comparison against the GCN is fair only under this protocol.
    """
    result = rules_in_sample(features, labels, threshold)
    correct = sum(
        1 for eid, truth in labels.items()
        if eid in features and assess(features[eid], RULE_WEIGHTS, threshold).flagged == truth
    )
    lo, hi = _wilson(correct, len(labels))
    return {**result, "protocol": "leave-one-out (identical: no fitted parameters)",
            "accuracy": round(correct / len(labels), 4) if labels else 0.0, "accuracy_95ci": [lo, hi]}


def compare(
    graph: OwnershipGraph,
    features: dict[str, StructuralFeatures],
    labels: dict[str, bool],
    threshold: float = DEFAULT_THRESHOLD,
    gcn_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gcn_kwargs = gcn_kwargs or {"epochs": 300, "hidden_dim": 8}
    rules_loo = rules_leave_one_out(features, labels, threshold)
    gcn_is = gcn_in_sample(graph, labels, **gcn_kwargs)
    gcn_loo = gcn_leave_one_out(graph, labels, **gcn_kwargs)
    return {
        "labelled_structures": len(labels),
        "positive_labels": sum(1 for v in labels.values() if v),
        "rules": {"in_sample": rules_in_sample(features, labels, threshold), "leave_one_out": rules_loo},
        "gcn": {"in_sample": gcn_is, "leave_one_out": gcn_loo},
        "verdict": {
            "rules_f1": rules_loo["f1"],
            "gcn_f1_in_sample": gcn_is["f1"],
            "gcn_f1_leave_one_out": gcn_loo["f1"],
            "gcn_memorisation_gap": round(gcn_is["f1"] - gcn_loo["f1"], 4),
            "reading": (
                f"The GCN carries {gcn_is.get('parameters')} parameters against {len(labels)} labels. Its in-sample "
                "score is what memorisation looks like; the leave-one-out score is what it actually knows. The "
                "engineered features win here not because feature engineering beats representation learning, but "
                "because eleven labelled structures is not a training set."
            ),
        },
    }
