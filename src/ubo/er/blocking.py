"""Candidate generation.

426 records is 90,525 pairs. That is cheap. Six million records is eighteen
trillion pairs, and no scorer runs on that, so entity resolution at register
scale is a blocking problem first and a matching problem second.

The thing to watch is the trade: reduction ratio measures how much work you
avoided, candidate recall measures how much truth you threw away doing it, and
only reporting the first is how a pipeline quietly loses half its matches. Both
are computed here and both go in the report.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from ..config import DEFAULT_BLOCKING, BlockingConfig
from ..registers.loaders import Record
from .normalize import (
    address_tokens,
    normalize_identifier,
    normalize_jurisdiction,
    normalize_name,
    phonetic_key,
    significant_tokens,
)


@dataclass
class BlockingReport:
    n_records: int
    n_all_pairs: int
    n_candidate_pairs: int
    keys_per_record: float
    blocks: int
    dropped_blocks: int
    dropped_pairs_estimate: int
    by_key_type: dict[str, int]

    @property
    def reduction_ratio(self) -> float:
        return 1.0 - (self.n_candidate_pairs / self.n_all_pairs) if self.n_all_pairs else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.n_records,
            "all_pairs": self.n_all_pairs,
            "candidate_pairs": self.n_candidate_pairs,
            "reduction_ratio": round(self.reduction_ratio, 6),
            "keys_per_record": round(self.keys_per_record, 2),
            "blocks": self.blocks,
            "oversized_blocks_dropped": self.dropped_blocks,
            "pairs_dropped_by_size_cap": self.dropped_pairs_estimate,
            "candidates_by_key_type": self.by_key_type,
        }


def address_document_frequency(records: Sequence[Record]) -> dict[str, int]:
    df: dict[str, int] = defaultdict(int)
    for record in records:
        for token in address_tokens(record.address):
            df[token] += 1
    return dict(df)


def record_keys(
    record: Record,
    cfg: BlockingConfig = DEFAULT_BLOCKING,
    address_df: dict[str, int] | None = None,
) -> list[tuple[str, str]]:
    """Blocking keys for one record, as (key_type, key) pairs.

    Every key is prefixed with the entity type. A person and a company are never
    the same entity, and letting them share a block wastes the budget on pairs
    that the scorer rejects on the first feature.
    """
    keys: list[tuple[str, str]] = []
    kind = record.entity_type[:1]

    # Aliases are blocked on as well as the primary name. A sanctions record
    # whose romanised alias is the only form that matches is useless if only its
    # native-script primary name ever reaches a block.
    for form in dict.fromkeys([record.name, *record.aliases]):
        if not form:
            continue
        normalized = normalize_name(form)
        tokens = significant_tokens(form)

        if cfg.use_name_tokens and tokens:
            size = min(cfg.token_key_size, len(tokens))
            for combo in combinations(sorted(set(tokens)), size):
                keys.append(("name_tokens", f"{kind}|" + "+".join(combo)))

        if cfg.use_prefix and normalized:
            keys.append(("prefix", f"{kind}|{normalized.replace(' ', '')[: cfg.prefix_length]}"))

        if cfg.use_phonetic:
            phonetic = phonetic_key(form)
            if phonetic:
                keys.append(("phonetic", f"{kind}|{phonetic}"))

    if cfg.use_address:
        # Rarest-first, and only tokens rare enough to discriminate. A shared
        # registered office is common by design - corporate service providers
        # register thousands of companies at one address - so an unfiltered
        # address key is the single largest source of useless candidate pairs.
        df = address_df or {}
        addr = [t for t in address_tokens(record.address) if len(t) > 3 and df.get(t, 1) <= cfg.max_address_df]
        for token in sorted(addr, key=lambda t: (df.get(t, 1), t))[:2]:
            keys.append(("address", f"{kind}|{token}"))

    if cfg.use_identifier:
        ident = normalize_identifier(record.identifier)
        if ident:
            keys.append(("identifier", f"{kind}|{ident}"))
        jurisdiction = normalize_jurisdiction(record.jurisdiction)
        if jurisdiction and record.birth_date:
            keys.append(("dob_jurisdiction", f"{kind}|{jurisdiction}|{record.birth_date[:4]}"))

    return keys


def build_blocks(
    records: Sequence[Record], cfg: BlockingConfig = DEFAULT_BLOCKING
) -> tuple[dict[tuple[str, str], list[str]], dict[str, list[tuple[str, str]]]]:
    address_df = address_document_frequency(records) if cfg.use_address else {}
    blocks: dict[tuple[str, str], list[str]] = defaultdict(list)
    per_record: dict[str, list[tuple[str, str]]] = {}
    for record in records:
        keys = record_keys(record, cfg, address_df)
        per_record[record.record_id] = keys
        for key in keys:
            blocks[key].append(record.record_id)
    return blocks, per_record


def candidate_pairs(
    records: Sequence[Record], cfg: BlockingConfig = DEFAULT_BLOCKING
) -> tuple[set[tuple[str, str]], BlockingReport]:
    blocks, per_record = build_blocks(records, cfg)

    pairs: set[tuple[str, str]] = set()
    by_key_type: dict[str, int] = defaultdict(int)
    dropped_blocks = 0
    dropped_pairs = 0

    for (key_type, _key), members in blocks.items():
        if len(members) < 2:
            continue
        if len(members) > cfg.max_block_size:
            # A block this large is a token that is not discriminative - a shared
            # service address, or a legal form that survived normalisation.
            # Emitting it would cost more pairs than the whole rest of the run.
            dropped_blocks += 1
            dropped_pairs += len(members) * (len(members) - 1) // 2
            continue
        before = len(pairs)
        for a, b in combinations(sorted(members), 2):
            pairs.add((a, b))
        by_key_type[key_type] += len(pairs) - before

    n = len(records)
    report = BlockingReport(
        n_records=n,
        n_all_pairs=n * (n - 1) // 2,
        n_candidate_pairs=len(pairs),
        keys_per_record=sum(len(v) for v in per_record.values()) / n if n else 0.0,
        blocks=len(blocks),
        dropped_blocks=dropped_blocks,
        dropped_pairs_estimate=dropped_pairs,
        by_key_type=dict(sorted(by_key_type.items())),
    )
    return pairs, report


def true_pairs(clusters: dict[str, Iterable[str]]) -> set[tuple[str, str]]:
    """Every within-cluster pair from the ground-truth clustering."""
    out: set[tuple[str, str]] = set()
    for members in clusters.values():
        members = sorted(set(members))
        for a, b in combinations(members, 2):
            out.add((a, b))
    return out


def candidate_recall(candidates: set[tuple[str, str]], truth: set[tuple[str, str]]) -> dict[str, Any]:
    """The ceiling on everything downstream.

    A true pair that blocking never proposes cannot be recovered by any scorer,
    any adjudicator, or any amount of threshold tuning. This number is therefore
    the first thing to look at when recall disappoints, and the misses are
    listed so the failing key can be identified rather than guessed at.
    """
    if not truth:
        return {"candidate_recall": float("nan"), "true_pairs": 0, "missed": 0, "missed_examples": []}
    found = candidates & truth
    missed = sorted(truth - candidates)
    return {
        "candidate_recall": round(len(found) / len(truth), 4),
        "true_pairs": len(truth),
        "found": len(found),
        "missed": len(missed),
        "missed_examples": missed[:10],
    }
