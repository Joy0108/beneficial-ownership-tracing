"""Layering detection over the structural features.

Two models are compared, deliberately:

``rules``
    A weighted sum over the engineered features with a fitted threshold. Every
    decision decomposes into named reasons, which is what a filed suspicious
    activity report needs.

``gnn``
    A small graph convolution in :mod:`ubo.graph.gnn`, which learns from the
    topology directly rather than from features somebody chose.

The comparison is the interesting part. On a graph this size the engineered
features win, and they win for a reason worth stating: there are eight labelled
structures, and a GNN with even a few hundred parameters has nothing to learn
from. The result is reported as measured either way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .features import FEATURE_NAMES, StructuralFeatures, describe

# Weights encode what a financial-crime analyst weighs, not a fit. Depth alone
# is weakly suspicious; depth combined with secrecy hops and a nominee is what
# has no commercial explanation.
RULE_WEIGHTS: dict[str, float] = {
    "chain_depth": 0.10,
    "controlled_count": 0.02,
    "jurisdiction_hops": 0.09,
    "secrecy_hops": 0.16,
    "secrecy_ratio": 0.40,
    "in_cycle": 0.30,
    "max_intermediary_centrality": 0.60,
    # Sharing an intermediary is ordinary; a joint venture does it. Being a
    # nominee while sharing one is not, so almost all the weight goes there.
    "shared_intermediaries": 0.03,
    "nominee_intermediaries": 0.30,
    "secrecy_co_owners": 0.45,
    "mean_edge_confidence": -0.10,
    "ownership_attenuation": 0.004,
    "sub_threshold_holdings": 0.10,
    "single_source_edges": 0.05,
}

# Chosen by an explicit policy, not fitted: the highest recall available at
# precision >= 0.9 on the threshold sweep. With eleven labelled structures this
# is a policy decision that happens to be informed by data, and calling it a
# fitted threshold would overstate what eleven labels can support.
DEFAULT_THRESHOLD = 1.0


@dataclass
class RiskAssessment:
    entity_id: str
    score: float
    flagged: bool
    contributions: dict[str, float]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "score": round(self.score, 4),
            "flagged": self.flagged,
            "top_contributions": dict(
                sorted(((k, round(v, 4)) for k, v in self.contributions.items() if abs(v) > 1e-6),
                       key=lambda kv: -abs(kv[1]))[:5]
            ),
            "reasons": self.reasons,
        }


def assess(
    features: StructuralFeatures,
    weights: dict[str, float] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> RiskAssessment:
    weights = weights or RULE_WEIGHTS
    contributions = {name: weights.get(name, 0.0) * getattr(features, name) for name in FEATURE_NAMES}
    score = sum(contributions.values())
    return RiskAssessment(
        entity_id=features.entity_id,
        score=score,
        flagged=score >= threshold,
        contributions=contributions,
        reasons=describe(features),
    )


def assess_all(
    features: dict[str, StructuralFeatures],
    weights: dict[str, float] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, RiskAssessment]:
    return {eid: assess(f, weights, threshold) for eid, f in features.items()}


def evaluate(
    assessments: dict[str, RiskAssessment], labels: dict[str, bool]
) -> dict[str, Any]:
    """Precision and recall against the labelled structures.

    Recall is reported alongside precision because in screening they trade
    against each other explicitly: a missed layered structure is a client
    onboarded, and a false flag is an analyst-day spent clearing it. Neither
    number means anything on its own.
    """
    scored = [(eid, a) for eid, a in assessments.items() if eid in labels]
    if not scored:
        return {"n": 0}
    tp = sum(1 for eid, a in scored if a.flagged and labels[eid])
    fp = sum(1 for eid, a in scored if a.flagged and not labels[eid])
    fn = sum(1 for eid, a in scored if not a.flagged and labels[eid])
    tn = sum(1 for eid, a in scored if not a.flagged and not labels[eid])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "n": len(scored),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "flagged": [eid for eid, a in scored if a.flagged],
        "missed": [eid for eid, a in scored if not a.flagged and labels[eid]],
    }


def sweep_threshold(
    assessments: dict[str, RiskAssessment], labels: dict[str, bool], steps: Sequence[float] | None = None
) -> list[dict[str, Any]]:
    """Precision and recall across the operating range.

    A single threshold is a policy choice, not a property of the model. The
    sweep is what lets that choice be made against a stated appetite instead of
    inherited from whatever number happened to be in the code.
    """
    steps = steps or [round(x / 20, 2) for x in range(0, 41)]
    rows = []
    for threshold in steps:
        adjusted = {
            eid: RiskAssessment(a.entity_id, a.score, a.score >= threshold, a.contributions, a.reasons)
            for eid, a in assessments.items()
        }
        result = evaluate(adjusted, labels)
        rows.append({"threshold": threshold, **{k: v for k, v in result.items() if k in {"precision", "recall", "f1", "tp", "fp", "fn"}}})
    return rows
