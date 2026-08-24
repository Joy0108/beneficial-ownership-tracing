"""Due-diligence memo generation.

Two backends behind one interface, the same arrangement as the adjudicator: a
deterministic template that CI can gate on, and Claude for the version a human
would actually read. Both produce the same structure and both go through the
same citation and PEP checks afterwards, because the checks are the guarantee
and the generator is not.

Every factual sentence carries either a register citation - a statement id from
an actual filing - or a regulatory citation to a section a compliance officer
can look up. A memo that asserts something with neither is rejected by the
critic rather than published with a caveat.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..er.resolve import ResolvedEntity
from ..graph.build import OwnershipGraph  # noqa: F401  (used in type hints below)
from ..graph.features import StructuralFeatures, describe
from ..graph.patterns import RiskAssessment
from . import pep

CITATION = re.compile(r"\[(?:cite|reg|stmt):([A-Za-z0-9_.:/\-]+)\]")

SYSTEM_PROMPT = """You draft beneficial ownership due-diligence memos for a financial crime team.

Non-negotiable rules:
- Every factual claim ends with a citation: [stmt:<statement_id>] for a register filing, [reg:<section_id>] for regulatory guidance.
- Assert nothing the evidence does not carry. Ownership percentages, jurisdictions and dates come from the statements, never from inference.
- A politically exposed person match requires enhanced due diligence under FATF Recommendation 12. It is not an adverse finding and it is never a reason to decline a relationship. Do not describe a PEP as adverse, criminal, corrupt or suspicious.
- A long ownership chain is not by itself evidence of concealment. Say what makes this structure different, or say that nothing does.
- You recommend; you do not decide. The memo ends at a recommendation for human review."""


@dataclass
class Memo:
    entity_id: str
    text: str
    citations: list[str] = field(default_factory=list)
    backend: str = "template"
    pep_check: dict[str, Any] = field(default_factory=dict)

    def cited(self) -> list[str]:
        return CITATION.findall(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "backend": self.backend,
            "citations": self.cited(),
            "pep_check": self.pep_check,
            "text": self.text,
        }


class MemoWriter(Protocol):
    name: str

    def write(self, context: dict[str, Any]) -> Memo: ...


class TemplateMemoWriter:
    name = "template"

    def write(self, context: dict[str, Any]) -> Memo:
        entity: ResolvedEntity = context["entity"]
        features: StructuralFeatures = context["features"]
        assessment: RiskAssessment = context["assessment"]
        guidance: Sequence[dict[str, Any]] = context.get("guidance", [])
        chain: Sequence[dict[str, Any]] = context.get("chain", [])

        lines = [
            f"# Beneficial ownership review: {entity.canonical_name}",
            "",
            f"**Entity** {entity.entity_id} ({entity.entity_type})  |  "
            f"**Jurisdictions** {', '.join(entity.jurisdictions) or 'not recorded'}  |  "
            f"**Registers** {', '.join(entity.sources)}",
            "",
            "## Identity resolution",
            "",
            f"This entity is a merge of {len(entity.record_ids)} register records across "
            f"{len(entity.sources)} sources, the weakest supporting link scoring "
            f"{entity.weakest_link:.2f}. The records are:",
            "",
        ]
        for prov in entity.provenance:
            lines.append(f"- `{prov['record_id']}` ({prov['source']}) recorded as \"{prov['name_as_recorded']}\"")

        lines += ["", "## Ownership and control", ""]
        if chain:
            for step in chain:
                pct = f"{step['share_percent']:.0f}%" if step["share_percent"] else "no percentage recorded"
                stmt = step["statements"][0]["statement_id"] if step["statements"] else "unknown"
                lines.append(
                    f"- {step['parent_name']} holds {pct} of {step['child_name']} "
                    f"({step['child_jurisdiction'] or 'jurisdiction not recorded'}), asserted by "
                    f"{step['statements'][0]['source'] if step['statements'] else 'unknown'} [stmt:{stmt}]"
                )
        else:
            lines.append("- No ownership or control statement in the ingested registers names this entity as a holder.")

        lines += ["", "## Structural assessment", ""]
        reasons = describe(features)
        if reasons:
            lines += [f"- {reason}" for reason in reasons]
        else:
            lines.append("- No structural risk indicator fired on this entity.")
        lines += [
            "",
            f"Composite structural score {assessment.score:.2f} against a review threshold of "
            f"{context.get('threshold', 1.0):.2f}; the assessment is "
            f"{'above' if assessment.flagged else 'below'} the threshold.",
        ]

        if entity.is_sanctioned:
            lines += [
                "",
                "## Sanctions",
                "",
                "One or more source registers list this entity under a sanctions programme. Where blocked persons "
                "hold fifty percent or more in aggregate, directly or indirectly, the held entity is itself blocked "
                "whether or not it appears on the list [reg:FFIEC-SANCTIONS].",
            ]

        if entity.is_pep:
            position = entity.pep_position or "a public position"
            lines += [
                "",
                "## Politically exposed person status",
                "",
                f"A source register records this individual as holding {position}. Under FATF Recommendation 12 this "
                "requires enhanced due diligence: senior management approval, establishment of source of wealth and "
                "source of funds, and enhanced ongoing monitoring [reg:FATF-R12]. It is not an adverse finding and "
                "it does not on its own support declining the relationship [reg:FFIEC-EDD].",
            ]

        if guidance:
            lines += ["", "## Applicable guidance", ""]
            for section in guidance:
                lines.append(f"- **{section['id']}** {section['title']} [reg:{section['id']}]")

        lines += [
            "",
            "## Recommendation",
            "",
            (
                "Escalate for analyst review before any decision. The structural indicators above are grounds for "
                "enhanced due diligence, not a conclusion [reg:FFIEC-EDD]."
                if assessment.flagged
                else "No escalation indicated by the structural review. Standard due diligence applies [reg:FFIEC-CDD]."
            ),
            "",
            "This memo is a recommendation. The decision is recorded by the reviewing analyst.",
        ]

        text = "\n".join(lines)
        text, check = pep.enforce(text, entity.is_pep)
        return Memo(entity.entity_id, text, backend=self.name, pep_check=check.to_dict())


class ClaudeMemoWriter:  # pragma: no cover - requires credentials
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 4000):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.fallback = TemplateMemoWriter()

    def write(self, context: dict[str, Any]) -> Memo:
        import json

        entity: ResolvedEntity = context["entity"]
        payload = {
            "entity": entity.to_dict(),
            "ownership_chain": context.get("chain", []),
            "structural_features": context["features"].to_dict(),
            "assessment": context["assessment"].to_dict(),
            "regulatory_sections": context.get("guidance", []),
        }
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2, default=str)}],
            )
            text = "".join(b.text for b in response.content if b.type == "text").strip()
        except Exception:
            memo = self.fallback.write(context)
            memo.backend = "template (anthropic unavailable)"
            return memo

        # The gate runs on the model's output exactly as it runs on the
        # template's. A prompt asking for correct PEP framing is a request; this
        # is the part that makes it a property of the system.
        text, check = pep.enforce(text, entity.is_pep)
        return Memo(entity.entity_id, text, backend=self.name, pep_check=check.to_dict())


def build_writer(backend: str | None = None, model: str = "claude-opus-5") -> MemoWriter:
    backend = backend or os.environ.get("UBO_LLM", "deterministic")
    if backend in {"anthropic", "claude"}:
        return ClaudeMemoWriter(model=model)
    return TemplateMemoWriter()


def chain_view(graph: OwnershipGraph, root: str, max_depth: int = 12) -> list[dict[str, Any]]:
    """The deepest control chain, flattened for the memo with provenance intact."""
    paths = graph.paths(root, max_depth)
    if not paths:
        return []
    deepest = max(paths, key=len)
    out = []
    for edge in deepest:
        parent = graph.entities.get(edge.parent)
        child = graph.entities.get(edge.child)
        out.append({
            "parent": edge.parent,
            "parent_name": parent.canonical_name if parent else edge.parent,
            "child": edge.child,
            "child_name": child.canonical_name if child else edge.child,
            "child_jurisdiction": graph.jurisdiction_of(edge.child),
            "interest_type": edge.interest_type,
            "share_percent": edge.share_percent,
            "confidence": edge.confidence,
            "statements": edge.statements,
        })
    return out


def verify_citations(memo: Memo, valid_statements: set[str], valid_sections: set[str]) -> dict[str, Any]:
    """Every citation must resolve to a real statement or a real section."""
    cited = memo.cited()
    unresolved = [c for c in cited if c not in valid_statements and c not in valid_sections]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", memo.text) if len(s.split()) >= 8 and not s.startswith(("#", "-", "*", "|"))]
    uncited = [s for s in sentences if not CITATION.search(s)]
    return {
        "citations": len(cited),
        "resolved": len(cited) - len(unresolved),
        "resolution_rate": round((len(cited) - len(unresolved)) / len(cited), 4) if cited else 1.0,
        "unresolved": unresolved,
        "uncited_claim_sentences": len(uncited),
    }
