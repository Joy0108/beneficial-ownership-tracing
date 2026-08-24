"""Adjudication of the borderline band.

Most candidate pairs are decided by the score alone: a clear match or a clear
reject. The band in between is small and expensive - it is where a human analyst
would actually spend time, and where a cheap threshold move trades false merges
against missed links with no way to tell which you are getting.

Two backends, same interface:

``deterministic``
    A rule cascade over the same features the scorer produced, applied in the
    order an analyst would apply them. Free, reproducible, and what CI runs.

``anthropic``
    Claude, given the two records and asked to judge with a reason. Selected
    with ``UBO_LLM=anthropic``.

Adjudication never silently upgrades a pair: every decision carries the rule or
the model reason that produced it, and both are written to the audit trail.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import DEFAULT_ADJUDICATION, AdjudicationConfig
from ..registers.loaders import Record
from .normalize import normalize_name, significant_tokens
from .scoring import PairScore, token_containment

SYSTEM_PROMPT = """You adjudicate borderline entity-resolution pairs for a beneficial ownership screening system.

You are given two register records and the features a scorer computed for them. Decide whether they refer to the same real-world entity.

Rules:
- A conflict in birth date, or in jurisdiction of incorporation, is disqualifying on its own.
- Two unrelated companies frequently share a trading name in different jurisdictions. Name similarity alone is never sufficient for a company.
- A shared registered office is weak evidence: corporate service providers register thousands of companies at one address.
- Transliteration variants of the same personal name are the same name.
- When the evidence is genuinely balanced, answer "reject". A false merge silently attributes one party's holdings to another, and no downstream stage can detect it.

Answer with JSON only: {"decision": "match" | "reject", "reason": "<one sentence>", "confidence": <0..1>}"""


@dataclass
class Adjudication:
    left: str
    right: str
    decision: str  # match | reject
    reason: str
    confidence: float
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "decision": self.decision,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "backend": self.backend,
        }


class Adjudicator(Protocol):
    name: str

    def judge(self, left: Record, right: Record, score: PairScore) -> Adjudication: ...


class RuleAdjudicator:
    """An analyst's checklist, in the order an analyst applies it."""

    name = "deterministic"

    def judge(self, left: Record, right: Record, score: PairScore) -> Adjudication:
        f = score.features

        if f.get("birth_date_match", 0.0) < 0:
            return self._out(score, "reject", "the two records give conflicting birth dates", 0.95)

        if f.get("identifier_match", 0.0) == 1.0:
            return self._out(score, "match", "both records carry the same registered identifier", 0.99)

        if f.get("jurisdiction_match", 0.0) < 0 and left.entity_type == "company":
            return self._out(
                score, "reject",
                "same trading name in two different jurisdictions, which is common and not evidence of identity", 0.85)

        if left.entity_type == "person":
            # For a person, a corroborating birth date is what separates a real
            # match from a common-name collision.
            if f.get("birth_date_match", 0.0) >= 0.8 and f.get("name_jaro", 0.0) >= 0.85:
                return self._out(score, "match", "names agree and both registers give the same birth date", 0.92)
            if f.get("phonetic_match", 0.0) == 1.0 and f.get("name_jaro", 0.0) >= 0.88:
                return self._out(score, "match", "transliteration variants of the same name", 0.80)
            if f.get("name_jaro", 0.0) >= 0.94 and f.get("jurisdiction_match", 0.0) > 0:
                return self._out(score, "match", "near-identical name and the same nationality", 0.75)
            return self._out(score, "reject", "name similarity alone, with nothing corroborating it", 0.70)

        # Companies.
        containment = token_containment(left.name, right.name)
        if containment >= 0.99 and f.get("jurisdiction_match", 0.0) > 0:
            return self._out(
                score, "match",
                "one name is the other with a legal form or suffix, in the same jurisdiction", 0.88)
        if f.get("name_jaro", 0.0) >= 0.92 and f.get("address_overlap", 0.0) >= 0.5:
            return self._out(score, "match", "names agree and the registered addresses overlap", 0.82)
        if f.get("name_jaro", 0.0) >= 0.96 and f.get("jurisdiction_match", 0.0) > 0:
            return self._out(score, "match", "near-identical legal name in the same jurisdiction", 0.78)
        return self._out(score, "reject", "the evidence is balanced, so the pair is left unmerged", 0.60)

    @staticmethod
    def _out(score: PairScore, decision: str, reason: str, confidence: float) -> Adjudication:
        return Adjudication(score.left, score.right, decision, reason, confidence, "deterministic")


class ClaudeAdjudicator:  # pragma: no cover - requires credentials
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 1000):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.fallback = RuleAdjudicator()

    def judge(self, left: Record, right: Record, score: PairScore) -> Adjudication:
        payload = {
            "left": left.to_dict(),
            "right": right.to_dict(),
            "features": {k: round(v, 3) for k, v in score.features.items()},
            "score": round(score.score, 3),
        }
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}],
            )
            text = "".join(b.text for b in response.content if b.type == "text")
            parsed = json.loads(text[text.index("{") : text.rindex("}") + 1])
            decision = "match" if str(parsed.get("decision")).lower() == "match" else "reject"
            return Adjudication(
                score.left, score.right, decision,
                str(parsed.get("reason", ""))[:300], float(parsed.get("confidence", 0.5)), "anthropic",
            )
        except Exception as exc:
            # A screening pipeline cannot stall on an API error, and it must not
            # silently upgrade the pair either. Fall back to the rules and say so.
            fallback = self.fallback.judge(left, right, score)
            fallback.backend = f"deterministic (anthropic failed: {type(exc).__name__})"
            return fallback


def build_adjudicator(cfg: AdjudicationConfig = DEFAULT_ADJUDICATION) -> Adjudicator:
    backend = cfg.backend or os.environ.get("UBO_LLM", "deterministic")
    if backend in {"anthropic", "claude"}:
        return ClaudeAdjudicator(model=cfg.anthropic_model)
    return RuleAdjudicator()


def adjudicate_all(
    records: dict[str, Record],
    scores: Sequence[PairScore],
    adjudicator: Adjudicator | None = None,
    cfg: AdjudicationConfig = DEFAULT_ADJUDICATION,
) -> tuple[list[Adjudication], dict[str, Any]]:
    judge = adjudicator or build_adjudicator(cfg)
    borderline = [s for s in scores if s.decision == "review"]
    out: list[Adjudication] = []
    for score in borderline[: cfg.max_adjudications]:
        out.append(judge.judge(records[score.left], records[score.right], score))

    stats = {
        "backend": judge.name,
        "reviewed": len(out),
        "borderline_total": len(borderline),
        "truncated": max(0, len(borderline) - cfg.max_adjudications),
        "upheld_as_match": sum(1 for a in out if a.decision == "match"),
        "rejected": sum(1 for a in out if a.decision == "reject"),
    }
    return out, stats


def summarise_name(record: Record) -> str:
    return f"{record.name} [{normalize_name(record.name)}] tokens={significant_tokens(record.name)}"
