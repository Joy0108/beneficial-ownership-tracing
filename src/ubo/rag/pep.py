"""PEP language enforcement.

A politically exposed person is not a wrongdoer. Recommendation 12 requires
enhanced due diligence, senior management approval, and source-of-wealth work;
it does not require refusal, and it does not license the word "adverse".

The distinction is not pedantry. A memo that renders a PEP match as an adverse
finding is wrong on the law, and it is the mechanism by which whole categories
of customer get de-risked - which FATF has said repeatedly is a failure of the
standard, not a conservative application of it. This module is the gate that
stops the generator from producing that sentence, and it is enforced on the
output rather than requested in a prompt, because a prompt is a request and a
gate is a guarantee.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# Words that make a factual statement into an accusation.
_ADVERSE_TERMS = (
    "adverse", "criminal", "corrupt", "corruption", "launder", "laundering", "illicit",
    "fraud", "fraudulent", "wrongdoing", "guilty", "perpetrator", "offender", "suspect",
)

# Language that treats the status itself as disqualifying.
_DISQUALIFYING = (
    r"\b(must|should|will)\s+(be\s+)?(declin|reject|refus|exit|terminat|offboard|close)",
    r"\bdo\s+not\s+onboard\b",
    r"\bnot\s+(?:be\s+)?acceptab\w*\b",
    r"\bprohibited\s+(?:customer|relationship)\b",
    r"\bblacklist",
)

_PEP_CONTEXT = re.compile(
    r"\b(pep|politically\s+exposed|senior\s+(?:political|government)\s+official|"
    r"minister|governor|state\s+official)\b",
    re.I,
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+")

REQUIRED_FRAMING = (
    "A politically exposed person match is a category that requires enhanced due diligence under FATF "
    "Recommendation 12 - senior management approval, source of wealth and source of funds, and enhanced "
    "ongoing monitoring. It is not an adverse finding and it is not a reason to decline the relationship."
)


@dataclass
class PepCheck:
    violations: list[dict[str, Any]] = field(default_factory=list)
    pep_mentioned: bool = False
    framing_present: bool = False

    @property
    def passed(self) -> bool:
        if not self.pep_mentioned:
            return True
        return not self.violations and self.framing_present

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "pep_mentioned": self.pep_mentioned,
            "framing_present": self.framing_present,
            "violations": self.violations,
        }


def check(text: str) -> PepCheck:
    result = PepCheck()
    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]

    for sentence in sentences:
        if not _PEP_CONTEXT.search(sentence):
            continue
        result.pep_mentioned = True
        low = sentence.lower()

        # "not an adverse finding" is the correct framing, not a violation, so
        # negated forms are excluded before the term match.
        for term in _ADVERSE_TERMS:
            for match in re.finditer(rf"\b{term}\w*\b", low):
                if _is_negated(low, match.start()):
                    continue
                result.violations.append({
                    "kind": "adverse_characterisation",
                    "term": match.group(0),
                    "sentence": sentence[:220],
                })
                break

        for pattern in _DISQUALIFYING:
            if re.search(pattern, low):
                result.violations.append({
                    "kind": "disqualifying_language",
                    "pattern": pattern,
                    "sentence": sentence[:220],
                })

    if result.pep_mentioned:
        low_all = text.lower()
        result.framing_present = (
            "enhanced due diligence" in low_all
            and ("not an adverse finding" in low_all or "not itself an adverse" in low_all or "does not" in low_all)
        )
    return result


def _is_negated(text: str, position: int, window: int = 40) -> bool:
    prefix = text[max(0, position - window) : position]
    return bool(re.search(r"\b(not|never|neither|no)\b[^.]{0,30}$", prefix))


def enforce(text: str, is_pep: bool) -> tuple[str, PepCheck]:
    """Add the required framing when a PEP is discussed and it is missing.

    Violations are never rewritten away. Silently editing an accusation out of a
    draft would hide the fact that the generator produced one, and the whole
    point of the gate is that the failure is visible in the audit trail.
    """
    result = check(text)
    if is_pep and not result.framing_present:
        text = f"{text.rstrip()}\n\n{REQUIRED_FRAMING}"
        result = check(text)
    return text, result


def scan_all(texts: Iterable[str]) -> dict[str, Any]:
    checks = [check(t) for t in texts]
    relevant = [c for c in checks if c.pep_mentioned]
    return {
        "documents": len(checks),
        "mentioning_pep": len(relevant),
        "passing": sum(1 for c in relevant if c.passed),
        "pass_rate": round(sum(1 for c in relevant if c.passed) / len(relevant), 4) if relevant else float("nan"),
        "violations": [v for c in checks for v in c.violations],
    }
