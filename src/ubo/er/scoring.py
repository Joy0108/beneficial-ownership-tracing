"""Pairwise scoring.

A weighted sum of interpretable features, not a black box. Screening decisions
are contestable: an analyst who rejects a match has to be able to see which
feature carried it, and a regulator asking why two entities were merged needs an
answer better than "the model said so".
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ..config import DEFAULT_SCORING, ScoringConfig
from ..registers.loaders import Record
from .normalize import (
    address_tokens,
    normalize_birth_date,
    normalize_identifier,
    normalize_jurisdiction,
    normalize_name,
    phonetic_key,
    significant_tokens,
)


@dataclass
class PairScore:
    left: str
    right: str
    score: float
    features: dict[str, float]
    decision: str  # match | review | reject

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "score": round(self.score, 4),
            "decision": self.decision,
            "features": {k: round(v, 4) for k, v in self.features.items()},
        }


# --- string similarity ------------------------------------------------------

@lru_cache(maxsize=200_000)
def jaro(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    window = max(len(a), len(b)) // 2 - 1
    if window < 0:
        window = 0

    a_flags = [False] * len(a)
    b_flags = [False] * len(b)
    matches = 0
    for i, ch in enumerate(a):
        lo = max(0, i - window)
        hi = min(i + window + 1, len(b))
        for j in range(lo, hi):
            if not b_flags[j] and b[j] == ch:
                a_flags[i] = b_flags[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i, flag in enumerate(a_flags):
        if not flag:
            continue
        while not b_flags[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1
    transpositions //= 2

    return (matches / len(a) + matches / len(b) + (matches - transpositions) / matches) / 3.0


@lru_cache(maxsize=200_000)
def jaro_winkler(a: str, b: str, prefix_weight: float = 0.1, max_prefix: int = 4) -> float:
    """Jaro with a prefix bonus.

    The bonus matters here because transliteration mangles endings far more
    often than beginnings: Kuznetsova/Kouznetsova diverge at position two, but
    Morozov/Morosov share their first four characters.
    """
    base = jaro(a, b)
    if base < 0.7:
        return base
    prefix = 0
    for x, y in zip(a[:max_prefix], b[:max_prefix], strict=False):
        if x != y:
            break
        prefix += 1
    return base + prefix * prefix_weight * (1 - base)


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(significant_tokens(a)), set(significant_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def token_containment(a: str, b: str) -> float:
    """Overlap relative to the *shorter* name.

    "Baltic Resource" against "Baltic Resource Holdings Group International"
    scores badly on Jaccard and well here, which is the right reading: registers
    truncate long names constantly.
    """
    ta, tb = set(significant_tokens(a)), set(significant_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def address_overlap(a: str, b: str) -> float:
    ta, tb = address_tokens(a), address_tokens(b)
    if not ta or not tb:
        return 0.0  # absence of evidence, scored as no evidence, not as conflict
    return len(ta & tb) / min(len(ta), len(tb))


def birth_date_agreement(a: str, b: str) -> float:
    """Compare at whatever precision both sides actually published.

    PSC publishes year and month only. Requiring an exact match against a full
    date from OFAC would reject every genuine person-to-person match between
    those two registers.
    """
    na, nb = normalize_birth_date(a), normalize_birth_date(b)
    if not na or not nb:
        return 0.0
    pa, pb = na.split("-"), nb.split("-")
    depth = min(len(pa), len(pb))
    if pa[:depth] != pb[:depth]:
        return -1.0  # an actual conflict, which is evidence *against*
    return {1: 0.4, 2: 0.8, 3: 1.0}[depth]


def name_forms(record: Record) -> list[str]:
    """Every name the register published for this record.

    OpenSanctions carries the native-script form alongside the romanised one,
    and matching only the primary name throws away the single best piece of
    evidence a sanctions record has. Scoring takes the best pair across all
    forms, because two records match if *any* of their names refer to the same
    entity, not only if the fields the loaders happened to pick first do.
    """
    forms = [record.name, *record.aliases]
    return [f for f in dict.fromkeys(forms) if f]


def _best_over_forms(left: Record, right: Record, fn) -> float:
    return max((fn(a, b) for a in name_forms(left) for b in name_forms(right)), default=0.0)


def score_pair(left: Record, right: Record, cfg: ScoringConfig = DEFAULT_SCORING) -> PairScore:
    if cfg.type_must_match and left.entity_type != right.entity_type:
        return PairScore(left.record_id, right.record_id, 0.0, {"type_mismatch": 1.0}, "reject")

    lj, rj = normalize_jurisdiction(left.jurisdiction), normalize_jurisdiction(right.jurisdiction)
    li, ri = normalize_identifier(left.identifier), normalize_identifier(right.identifier)

    features = {
        "name_jaro": _best_over_forms(left, right, lambda a, b: jaro_winkler(normalize_name(a), normalize_name(b))),
        "name_token_jaccard": _best_over_forms(
            left, right, lambda a, b: max(token_jaccard(a, b), 0.85 * token_containment(a, b))),
        "phonetic_match": 1.0 if any(
            phonetic_key(a) and phonetic_key(a) == phonetic_key(b)
            for a in name_forms(left) for b in name_forms(right)) else 0.0,
        "address_overlap": address_overlap(left.address, right.address),
        "jurisdiction_match": 1.0 if lj and rj and lj == rj else (-0.5 if lj and rj else 0.0),
        "birth_date_match": birth_date_agreement(left.birth_date, right.birth_date),
        "identifier_match": 1.0 if li and ri and li == ri else 0.0,
    }

    score = sum(cfg.weights.get(k, 0.0) * v for k, v in features.items())

    # A shared strong identifier settles it. Two records carrying the same LEI
    # are the same legal entity by construction, whatever the strings say.
    if features["identifier_match"] == 1.0:
        score = max(score, 0.95)
    # A birth-date conflict is disqualifying for a person no matter how well the
    # names match, because common names are exactly where this fails.
    if features["birth_date_match"] < 0:
        score = min(score, cfg.review_low - 0.01)

    score = max(0.0, min(1.0, score))
    if score >= cfg.review_high:
        decision = "match"
    elif score >= cfg.review_low:
        decision = "review"
    else:
        decision = "reject"
    return PairScore(left.record_id, right.record_id, score, features, decision)


def score_candidates(
    records: dict[str, Record], pairs: set[tuple[str, str]], cfg: ScoringConfig = DEFAULT_SCORING
) -> list[PairScore]:
    out = []
    for a, b in sorted(pairs):
        left, right = records.get(a), records.get(b)
        if left is None or right is None:
            continue
        out.append(score_pair(left, right, cfg))
    return out


def explain(score: PairScore, cfg: ScoringConfig = DEFAULT_SCORING) -> str:
    """Human-readable reason, ordered by contribution."""
    contributions = sorted(
        ((k, cfg.weights.get(k, 0.0) * v) for k, v in score.features.items()),
        key=lambda kv: -abs(kv[1]),
    )
    parts = [f"{k}={score.features[k]:.2f} (contributes {c:+.3f})" for k, c in contributions if abs(c) > 1e-9]
    return f"score {score.score:.3f} -> {score.decision}; " + ", ".join(parts)
