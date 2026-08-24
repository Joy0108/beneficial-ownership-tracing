"""Fitting the pairwise scorer, and the experiment that needs it.

The leakage argument in ``splits.py`` only bites once something is actually
fitted. With fixed hand-set weights there is nothing for a shared record to leak
*into*: the model has no capacity to memorise. So this module fits the feature
weights and the decision threshold on the train half by random search, which is
enough capacity to overfit a specific set of names, and the split experiment
then measures how much of the resulting test score is real.

Random search rather than gradient descent because the objective is F1 over a
threshold - piecewise constant, non-differentiable, and small enough that a few
hundred samples cover the useful region.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..config import DEFAULT_SCORING, ScoringConfig
from .blocking import true_pairs
from .scoring import PairScore
from .splits import cluster_level_split, evaluate_on, negatives_for, pair_level_split

FEATURES = (
    "name_jaro",
    "name_token_jaccard",
    "phonetic_match",
    "address_overlap",
    "jurisdiction_match",
    "birth_date_match",
    "identifier_match",
)


@dataclass
class FitResult:
    weights: dict[str, float]
    threshold: float
    train_f1: float
    trials: int

    def as_config(self, base: ScoringConfig = DEFAULT_SCORING) -> ScoringConfig:
        return ScoringConfig(
            name="fitted",
            weights=dict(self.weights),
            match_threshold=self.threshold,
            review_low=max(0.0, self.threshold - 0.14),
            review_high=min(1.0, self.threshold + 0.08),
            type_must_match=base.type_must_match,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "threshold": round(self.threshold, 4),
            "train_f1": round(self.train_f1, 4),
            "trials": self.trials,
        }


def rescore(features: dict[str, float], weights: dict[str, float]) -> float:
    """Recompute a pair score from cached features under new weights."""
    if features.get("type_mismatch"):
        return 0.0
    score = sum(weights.get(k, 0.0) * v for k, v in features.items())
    if features.get("identifier_match", 0.0) == 1.0:
        score = max(score, 0.95)
    if features.get("birth_date_match", 0.0) < 0:
        score = min(score, 0.2)
    return max(0.0, min(1.0, score))


def predict(
    feature_table: dict[tuple[str, str], dict[str, float]],
    weights: dict[str, float],
    threshold: float,
    restrict_to: Iterable[tuple[str, str]] | None = None,
) -> set[tuple[str, str]]:
    keys = feature_table.keys() if restrict_to is None else restrict_to
    return {k for k in keys if rescore(feature_table[k], weights) >= threshold}


def feature_table(scores: Sequence[PairScore]) -> dict[tuple[str, str], dict[str, float]]:
    return {(s.left, s.right): s.features for s in scores}


def fit(
    table: dict[tuple[str, str], dict[str, float]],
    train_positives: set[tuple[str, str]],
    train_negatives: set[tuple[str, str]],
    trials: int = 400,
    seed: int = 11,
) -> FitResult:
    rng = random.Random(seed)
    scope = train_positives | train_negatives
    scope = {k for k in scope if k in table}

    best = FitResult(dict(DEFAULT_SCORING.weights), DEFAULT_SCORING.match_threshold, -1.0, trials)
    for trial in range(trials):
        if trial == 0:
            weights = dict(DEFAULT_SCORING.weights)  # always evaluate the hand-set prior
        else:
            raw = {f: rng.random() for f in FEATURES}
            total = sum(raw.values()) or 1.0
            weights = {f: v / total for f, v in raw.items()}

        for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            predicted = predict(table, weights, threshold, scope)
            f1 = evaluate_on(predicted, train_positives, train_negatives)["f1"]
            if f1 > best.train_f1:
                best = FitResult(weights, threshold, f1, trials)
    return best


class TokenMemoriser:
    """A matcher with enough capacity to memorise, which is the point.

    Learned record-linkage models are not seven weights over hand-built
    features; they have token-level or character-level parameters, and that is
    where leakage lives. This one learns which normalised name tokens appeared
    in a matching pair during training and matches on them at test time.

    It is not a strawman. It is what a learned model does when a rare token is
    highly predictive - ``northwind`` co-occurring in two matched records is a
    genuinely strong feature, right up until the evaluation split lets the same
    ``northwind`` records appear on both sides.
    """

    def __init__(self, records: dict[str, Any], min_precision: float = 0.8):
        self.records = records
        self.min_precision = min_precision
        self.matched_tokens: set[str] = set()

    def _shared(self, a: str, b: str) -> set[str]:
        from .normalize import significant_tokens

        ra, rb = self.records.get(a), self.records.get(b)
        if ra is None or rb is None:
            return set()
        return set(significant_tokens(ra.name)) & set(significant_tokens(rb.name))

    def train(
        self, positives: Iterable[tuple[str, str]], negatives: Iterable[tuple[str, str]] = ()
    ) -> TokenMemoriser:
        """Keep the tokens that discriminate, not every token ever shared.

        Trained on positives alone the model fires on ``capital`` and
        ``trading`` and is useless. Negatives are what turn it into a real
        matcher - and a real matcher is what makes the split comparison fair.
        """
        pos_counts: dict[str, int] = {}
        neg_counts: dict[str, int] = {}
        for a, b in positives:
            for token in self._shared(a, b):
                pos_counts[token] = pos_counts.get(token, 0) + 1
        for a, b in negatives:
            for token in self._shared(a, b):
                neg_counts[token] = neg_counts.get(token, 0) + 1

        for token, pos in pos_counts.items():
            neg = neg_counts.get(token, 0)
            if pos / (pos + neg) >= self.min_precision:
                self.matched_tokens.add(token)
        return self

    def predict(self, pairs: Iterable[tuple[str, str]]) -> set[tuple[str, str]]:
        from .normalize import significant_tokens

        out = set()
        for a, b in pairs:
            ra, rb = self.records.get(a), self.records.get(b)
            if ra is None or rb is None or ra.entity_type != rb.entity_type:
                continue
            shared = set(significant_tokens(ra.name)) & set(significant_tokens(rb.name))
            if shared & self.matched_tokens:
                out.add((a, b))
        return out


def capacity_experiment(
    records: dict[str, Any],
    clusters: dict[str, Sequence[str]],
    candidates: set[tuple[str, str]],
    test_fraction: float = 0.4,
    seed: int = 7,
) -> dict[str, Any]:
    """The same memorising matcher, scored under both split strategies."""
    positives = true_pairs(clusters)
    out: dict[str, Any] = {}
    for split in (pair_level_split(clusters, test_fraction, seed), cluster_level_split(clusters, test_fraction, seed)):
        train_neg = negatives_for(split, candidates, positives, "train")
        model = TokenMemoriser(records).train(split.train_pairs, train_neg)
        test_pos = split.test_pairs
        test_neg = negatives_for(split, candidates, positives, "test")
        predicted = model.predict(test_pos | test_neg)
        out[split.name] = {
            "records_in_both_halves": len(split.shared_records),
            "learned_tokens": len(model.matched_tokens),
            "test": evaluate_on(predicted, test_pos, test_neg),
        }
    inflation = round(out["pair_level"]["test"]["f1"] - out["cluster_level"]["test"]["f1"], 4)
    out["verdict"] = {
        "pair_level_test_f1": out["pair_level"]["test"]["f1"],
        "cluster_level_test_f1": out["cluster_level"]["test"]["f1"],
        "inflation": inflation,
        "reading": (
            "Identical model, identical data, identical training budget. Only the split differs. "
            "A pair-level benchmark reports this matcher as far better than it is, because the tokens it "
            "memorised in training are sitting in the test set attached to the same records."
        ),
    }
    return out


def split_experiment(
    clusters: dict[str, Sequence[str]],
    scores: Sequence[PairScore],
    candidates: set[tuple[str, str]],
    trials: int = 400,
    test_fraction: float = 0.4,
    seed: int = 7,
) -> dict[str, Any]:
    """Fit on train, score on test, once per split strategy.

    Same data, same model class, same number of trials. The only difference is
    whether a record may appear on both sides of the split.
    """
    table = feature_table(scores)
    positives = true_pairs(clusters)
    out: dict[str, Any] = {}

    for split in (pair_level_split(clusters, test_fraction, seed), cluster_level_split(clusters, test_fraction, seed)):
        train_pos = split.train_pairs
        train_neg = negatives_for(split, candidates, positives, "train")
        test_pos = split.test_pairs
        test_neg = negatives_for(split, candidates, positives, "test")

        fitted = fit(table, train_pos, train_neg, trials=trials, seed=seed)
        test_scope = test_pos | test_neg
        predicted = predict(table, fitted.weights, fitted.threshold, {k for k in test_scope if k in table})

        out[split.name] = {
            **split.to_dict(),
            "fit": fitted.to_dict(),
            "test": evaluate_on(predicted, test_pos, test_neg),
        }

    pair_f1 = out["pair_level"]["test"]["f1"]
    cluster_f1 = out["cluster_level"]["test"]["f1"]
    out["verdict"] = {
        "pair_level_test_f1": pair_f1,
        "cluster_level_test_f1": cluster_f1,
        "inflation": round(pair_f1 - cluster_f1, 4),
        "records_shared_between_halves": out["pair_level"]["records_in_both_halves"],
        "reading": (
            "Both numbers come from the same features and the same search budget. The pair-level split lets "
            f"{out['pair_level']['records_in_both_halves']} records appear in both halves, so weights tuned on train "
            "have already seen the exact strings they are tested on. The gap is how much a pair-level benchmark "
            "would overstate this matcher on entities it has never encountered."
        ),
    }
    return out
