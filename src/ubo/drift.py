"""Drift monitoring on the daily register deltas.

The distribution this pipeline sees moves for real reasons, not synthetic ones.
OpenSanctions publishes daily; a designation round adds hundreds of entities
from one jurisdiction overnight. GLEIF publishes daily golden-copy files; a
national registrar's bulk load changes the legal-form mix in a single day. Both
shift the input distribution without any code changing, and both change what the
blocking keys and the scorer see.

What is monitored is chosen to match how this pipeline actually fails:

* **Input distribution** - jurisdiction and source mix, name-length and
  token-count distributions. Population Stability Index, because it is the
  measure a model risk function will ask for.
* **Match-score distribution** - the pairwise scores are the model's output. A
  shift in their shape is the earliest signal that a threshold set months ago no
  longer means what it meant.
* **Volume** - the borderline band. A sharp rise in pairs needing adjudication
  is a cost signal and usually the first visible symptom of a bulk load.

Alerting on the score distribution rather than only on inputs matters: a drift
monitor that watches inputs alone reports green while the threshold quietly
drifts out of calibration underneath it.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .er.normalize import normalize_jurisdiction, significant_tokens
from .registers.loaders import Record

# Conventional PSI bands from credit risk model monitoring.
PSI_STABLE = 0.10
PSI_INVESTIGATE = 0.25


@dataclass
class DriftSignal:
    name: str
    statistic: str
    value: float
    threshold: float
    status: str  # stable | investigate | alert
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.name,
            "statistic": self.statistic,
            "value": round(self.value, 4),
            "threshold": self.threshold,
            "status": self.status,
            "detail": self.detail,
        }


def population_stability_index(
    baseline: Counter, current: Counter, epsilon: float = 1e-6
) -> tuple[float, dict[str, float]]:
    """PSI over a categorical distribution, with the per-category contributions.

    The contributions are returned because the aggregate number tells you that
    something moved and nothing else. Knowing that 80 percent of a PSI of 0.31
    comes from one jurisdiction turns an alert into a diagnosis.
    """
    categories = set(baseline) | set(current)
    base_total = sum(baseline.values()) or 1
    curr_total = sum(current.values()) or 1

    total = 0.0
    contributions: dict[str, float] = {}
    for category in categories:
        b = max(baseline.get(category, 0) / base_total, epsilon)
        c = max(current.get(category, 0) / curr_total, epsilon)
        contribution = (c - b) * math.log(c / b)
        contributions[category] = contribution
        total += contribution
    return total, dict(sorted(contributions.items(), key=lambda kv: -abs(kv[1])))


def _status(value: float, stable: float = PSI_STABLE, investigate: float = PSI_INVESTIGATE) -> str:
    if value < stable:
        return "stable"
    return "investigate" if value < investigate else "alert"


def jurisdiction_drift(baseline: Sequence[Record], current: Sequence[Record]) -> DriftSignal:
    base = Counter(normalize_jurisdiction(r.jurisdiction) or "unknown" for r in baseline)
    curr = Counter(normalize_jurisdiction(r.jurisdiction) or "unknown" for r in current)
    psi, contributions = population_stability_index(base, curr)
    return DriftSignal(
        "jurisdiction_mix", "psi", psi, PSI_INVESTIGATE, _status(psi),
        {"top_contributors": dict(list(contributions.items())[:5]), "new_jurisdictions": sorted(set(curr) - set(base))},
    )


def source_drift(baseline: Sequence[Record], current: Sequence[Record]) -> DriftSignal:
    base, curr = Counter(r.source for r in baseline), Counter(r.source for r in current)
    psi, contributions = population_stability_index(base, curr)
    return DriftSignal("source_mix", "psi", psi, PSI_INVESTIGATE, _status(psi),
                       {"top_contributors": dict(list(contributions.items())[:5])})


def name_shape_drift(baseline: Sequence[Record], current: Sequence[Record]) -> DriftSignal:
    """Token count per name, bucketed. Catches a bulk load with a different convention."""
    def buckets(records: Sequence[Record]) -> Counter:
        return Counter(str(min(len(significant_tokens(r.name)), 6)) for r in records)

    psi, contributions = population_stability_index(buckets(baseline), buckets(current))
    return DriftSignal("name_token_count", "psi", psi, PSI_INVESTIGATE, _status(psi),
                       {"top_contributors": dict(list(contributions.items())[:5])})


def score_drift(baseline_scores: Iterable[float], current_scores: Iterable[float], bins: int = 10) -> DriftSignal:
    """PSI over the match-score histogram.

    This is the signal that catches a decision threshold going stale. Inputs can
    look stable while the score distribution slides underneath a fixed cut-off.
    """
    def histogram(values: Iterable[float]) -> Counter:
        counter: Counter = Counter()
        for value in values:
            index = min(bins - 1, max(0, int(value * bins)))
            counter[str(index)] += 1
        return counter

    base, curr = histogram(baseline_scores), histogram(current_scores)
    psi, contributions = population_stability_index(base, curr)
    return DriftSignal("match_score_distribution", "psi", psi, PSI_INVESTIGATE, _status(psi),
                       {"bins": bins, "top_contributors": dict(list(contributions.items())[:5])})


def review_volume_drift(baseline_review: int, baseline_total: int, current_review: int, current_total: int) -> DriftSignal:
    base_rate = baseline_review / baseline_total if baseline_total else 0.0
    curr_rate = current_review / current_total if current_total else 0.0
    relative = abs(curr_rate - base_rate) / base_rate if base_rate else (1.0 if curr_rate else 0.0)
    return DriftSignal(
        "adjudication_volume", "relative_change", relative, 0.25,
        "stable" if relative < 0.15 else ("investigate" if relative < 0.25 else "alert"),
        {"baseline_rate": round(base_rate, 4), "current_rate": round(curr_rate, 4)},
    )


def run_monitors(
    baseline_records: Sequence[Record],
    current_records: Sequence[Record],
    baseline_scores: Sequence[float] = (),
    current_scores: Sequence[float] = (),
    review_counts: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    signals = [
        jurisdiction_drift(baseline_records, current_records),
        source_drift(baseline_records, current_records),
        name_shape_drift(baseline_records, current_records),
    ]
    if baseline_scores and current_scores:
        signals.append(score_drift(baseline_scores, current_scores))
    if review_counts:
        signals.append(review_volume_drift(*review_counts))

    worst = max((s.status for s in signals), key=lambda s: {"stable": 0, "investigate": 1, "alert": 2}[s])
    return {
        "baseline_records": len(baseline_records),
        "current_records": len(current_records),
        "overall_status": worst,
        "signals": [s.to_dict() for s in signals],
        "action": {
            "stable": "no action",
            "investigate": "review the top contributors before the next promotion",
            "alert": "hold promotion and re-baseline the decision thresholds",
        }[worst],
    }


def simulate_designation_round(
    records: Sequence[Record], jurisdiction: str = "RU", multiplier: int = 4
) -> list[Record]:
    """Reproduce the shape of a sanctions designation round for testing.

    Not a random perturbation: designation rounds are concentrated in one
    jurisdiction and arrive over a single day, which is exactly the shape a
    monitor tuned on gaussian noise fails to notice.
    """
    target = [r for r in records if normalize_jurisdiction(r.jurisdiction) == jurisdiction]
    return list(records) + [r for r in target for _ in range(multiplier)]
