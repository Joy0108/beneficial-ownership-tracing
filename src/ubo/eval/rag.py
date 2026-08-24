"""Regulatory retrieval evaluation.

Recall@5 rather than nDCG: a compliance memo cites a handful of sections and the
question is whether the right one is among them, not how it was ordered inside
the five. Precision is reported too, because retrieving all twenty sections
would score a perfect recall and be useless.

The PEP subset is scored separately and gated at 100 percent. Everything else in
this file is a quality metric; that one is a correctness requirement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import GOLDEN_PATH
from ..rag import pep
from ..rag.retrieve import RegulatoryIndex


def load_golden(path: Path = GOLDEN_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean(values) -> float:
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def evaluate_retrieval(index: RegulatoryIndex, golden: dict[str, Any], k: int = 5) -> dict[str, Any]:
    rows = []
    for q in golden["questions"]:
        hits = [h["id"] for h in index.search(q["question"], k)]
        primary = set(q["primary"])
        relevant = primary | set(q.get("secondary", []))
        rank = next((i + 1 for i, h in enumerate(hits) if h in primary), None)
        rows.append({
            "qid": q["qid"],
            "recall@k": len(primary & set(hits)) / len(primary) if primary else float("nan"),
            "any_primary@k": 1.0 if primary & set(hits) else 0.0,
            "precision@k": len(relevant & set(hits)) / k,
            "mrr": 1.0 / rank if rank else 0.0,
            "hits": hits,
            "missed_primary": sorted(primary - set(hits)),
        })
    return {
        "k": k,
        "questions": len(rows),
        f"recall@{k}": round(_mean(r["recall@k"] for r in rows), 4),
        f"any_primary@{k}": round(_mean(r["any_primary@k"] for r in rows), 4),
        f"precision@{k}": round(_mean(r["precision@k"] for r in rows), 4),
        "mrr": round(_mean(r["mrr"] for r in rows), 4),
        "rows": rows,
    }


def evaluate_pep_language(index: RegulatoryIndex, golden: dict[str, Any], k: int = 5) -> dict[str, Any]:
    """The PEP subset, checked two ways.

    First that retrieval surfaces the sections that establish the correct
    framing. Second, and separately, that the language gate rejects an adverse
    characterisation and accepts the correct one - because a system can retrieve
    the right guidance and still write the wrong sentence.
    """
    subset = [q for q in golden["questions"] if q.get("pep_language_test")]
    retrieval_rows = []
    for q in subset:
        hits = {h["id"] for h in index.search(q["question"], k)}
        retrieval_rows.append({
            "qid": q["qid"],
            "found_primary": bool(set(q["primary"]) & hits),
            "hits": sorted(hits),
        })

    probes = [
        ("The beneficial owner is a PEP, which is an adverse finding.", False),
        ("The customer is politically exposed and the relationship must be declined.", False),
        ("A PEP match indicates corruption risk on the part of the individual.", False),
        (
            "A source register records this individual as a PEP. Under FATF Recommendation 12 this requires "
            "enhanced due diligence: senior management approval, source of wealth and source of funds, and "
            "enhanced ongoing monitoring. It is not an adverse finding and it does not support declining the "
            "relationship.",
            True,
        ),
        (
            "The beneficial owner holds a ministerial position. Enhanced due diligence applies; this does not "
            "constitute an adverse finding.",
            True,
        ),
    ]
    probe_rows = []
    for text, should_pass in probes:
        result = pep.check(text)
        probe_rows.append({
            "expected_pass": should_pass,
            "actual_pass": result.passed,
            "correct": result.passed == should_pass,
            "violations": [v["kind"] for v in result.violations],
            "text": text[:90],
        })

    return {
        "retrieval_questions": len(subset),
        "retrieval_found_primary": sum(1 for r in retrieval_rows if r["found_primary"]),
        "retrieval_rate": round(sum(1 for r in retrieval_rows if r["found_primary"]) / len(subset), 4) if subset else float("nan"),
        "language_probes": len(probe_rows),
        "language_correct": sum(1 for r in probe_rows if r["correct"]),
        "language_rate": round(sum(1 for r in probe_rows if r["correct"]) / len(probe_rows), 4),
        "probe_detail": probe_rows,
        "rows": retrieval_rows,
    }


def ablate(golden: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
    """Lexical only, dense only, fused, fused plus rerank."""
    from dataclasses import replace

    from ..config import DEFAULT_RAG

    rows = []
    for name, cfg in [
        ("bm25 only", replace(DEFAULT_RAG, rerank=False)),
        ("dense only", replace(DEFAULT_RAG, rerank=False)),
        ("rrf fused", replace(DEFAULT_RAG, rerank=False)),
        ("rrf + rerank", DEFAULT_RAG),
    ]:
        index = RegulatoryIndex(cfg=cfg)
        if name == "bm25 only":
            index.search = lambda q, kk=None, _i=index: [  # type: ignore[method-assign]
                {**_i.by_id[s].to_dict(), "text": _i.by_id[s].text} for s, _ in _i.bm25.search(q, kk or k)
            ]
        elif name == "dense only":
            index.search = lambda q, kk=None, _i=index: [  # type: ignore[method-assign]
                {**_i.by_id[s].to_dict(), "text": _i.by_id[s].text} for s, _ in _i.dense.search(q, kk or k)
            ]
        result = evaluate_retrieval(index, golden, k)
        rows.append({"config": name, **{kk: vv for kk, vv in result.items() if kk != "rows"}})
    return rows
