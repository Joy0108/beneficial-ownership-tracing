"""Hybrid retrieval over the regulatory corpus.

Sections rather than sliding windows, because regulatory text is already
structured and a citation has to resolve to something a compliance officer can
look up. Splitting 31 CFR 1010.230 across two overlapping windows produces
citations to nothing.

BM25 and a corpus-fitted dense projection, fused with RRF and reranked. The
lexical half is not optional here: these queries are full of exact strings -
"twenty-five percent", "31 CFR 1010.230", "Recommendation 12" - and an
embedding blurs precisely the tokens that identify the right section.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DEFAULT_RAG, REGULATORY_DIR, RagConfig

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+")
_STOP = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the to was were will with this
    these those which what when where how why does do can could should would must may""".split()
)

# Regulatory text spells numbers out; queries do not, and vice versa. The
# threshold is the single most discriminative token in half this corpus, so
# "twenty-five percent" and "25%" have to reach the same block.
#
# The substitution runs on the *phrase* before tokenisation. Doing it per token
# does not work: "twenty-five" splits into "twenty" and "five" first, and by
# then the number is gone.
_NUMBER_PHRASES = [
    (re.compile(r"\btwenty[-\s]?five\b"), "twenty-five 25"),
    (re.compile(r"\b25\b"), "25 twenty-five"),
    (re.compile(r"\bfifty\b"), "fifty 50"),
    (re.compile(r"\b50\b"), "50 fifty"),
    (re.compile(r"\bthirty\b"), "thirty 30"),
    (re.compile(r"\b30\b"), "30 thirty"),
    (re.compile(r"\bfive\s+years\b"), "five 5 years"),
    (re.compile(r"\b5\s+years\b"), "5 five years"),
]

# Domain synonyms. Guidance and practitioners use different verbs for the same
# act - a bank "exits" a customer, FATF "terminates a business relationship" -
# and on a corpus this small there is not enough co-occurrence for a fitted
# projection to learn the equivalence. The list is short, one-directional per
# entry, and every entry is a term of art rather than a general synonym.
_SYNONYMS = {
    "exit": ["terminate", "terminating"],
    "exiting": ["terminating", "restricting"],
    "derisk": ["terminating", "restricting", "classes"],
    "derisking": ["terminating", "restricting", "classes"],
    "offboard": ["terminating"],
    "category": ["class", "classes"],
    "categories": ["classes"],
    "whole": ["entire"],
    "all": ["entire"],
    "decline": ["refuse", "declined"],
    "ubo": ["beneficial", "owner"],
    "shell": ["nominee"],
    "layering": ["concealment", "obscure"],
    "opaque": ["concealment"],
    "sar": ["suspicious", "report"],
    "cdd": ["customer", "due", "diligence"],
    "edd": ["enhanced", "due", "diligence"],
    "pep": ["politically", "exposed"],
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    for pattern, replacement in _NUMBER_PHRASES:
        text = pattern.sub(replacement, text)

    out: list[str] = []
    for raw in _TOKEN.findall(text):
        if raw in _STOP:
            continue
        out.append(raw)
        out.extend(_SYNONYMS.get(raw, ()))
    return out


@dataclass
class Section:
    id: str
    source: str
    title: str
    text: str

    @property
    def indexed_text(self) -> str:
        # The identifier and title are indexed with the body so a query naming
        # the section by number retrieves it directly.
        return f"{self.id} {self.title}. {self.text}"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "source": self.source, "title": self.title}


def load_sections(directory: Path = REGULATORY_DIR) -> list[Section]:
    path = directory / "guidance.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Section(id=s["id"], source=s["source"], title=s["title"], text=s["text"]) for s in payload["sections"]]


class BM25:
    def __init__(self, sections: Sequence[Section], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = [s.id for s in sections]
        self.freqs = [Counter(tokenize(s.indexed_text)) for s in sections]
        self.lengths = [sum(f.values()) for f in self.freqs]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.df: Counter = Counter()
        for f in self.freqs:
            self.df.update(f.keys())

    def idf(self, term: str) -> float:
        n, df = len(self.ids), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5)) if df else 0.0

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        terms = tokenize(query)
        scores: dict[int, float] = defaultdict(float)
        for term in set(terms):
            idf = self.idf(term)
            if idf <= 0:
                continue
            for i, freq in enumerate(self.freqs):
                tf = freq.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * (self.lengths[i] / self.avgdl if self.avgdl else 1))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.ids[kv[0]]))
        return [(self.ids[i], round(s, 6)) for i, s in ranked[:top_k]]


class DenseIndex:
    """TF-IDF reduced by truncated SVD. No model download, no GPU."""

    def __init__(self, sections: Sequence[Section], dim: int = 96):
        self.ids = [s.id for s in sections]
        docs = [tokenize(s.indexed_text) for s in sections]
        df = Counter()
        for d in docs:
            df.update(set(d))
        n = max(1, len(docs))
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in sorted(df)])

        matrix = np.zeros((len(docs), len(self.vocab)))
        for row, doc in enumerate(docs):
            for term, count in Counter(doc).items():
                matrix[row, self.vocab[term]] = 1.0 + math.log(count)
        matrix *= self.idf
        matrix = _l2(matrix)

        k = int(min(dim, min(matrix.shape) - 1)) or 1
        _, _, vt = np.linalg.svd(matrix, full_matrices=False)
        self.components = vt[:k].T
        self.vectors = _l2(matrix @ self.components)

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for term, count in Counter(tokenize(text)).items():
            idx = self.vocab.get(term)
            if idx is not None:
                vec[idx] = 1.0 + math.log(count)
        vec *= self.idf
        return _l2((vec @ self.components).reshape(1, -1))[0]

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        sims = self.vectors @ self._embed(query)
        order = np.argsort(-sims)[:top_k]
        return [(self.ids[i], round(float(sims[i]), 6)) for i in order]


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def rrf(rankings: Sequence[Sequence[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# ---------------------------------------------------------------------------
# Regime routing
# ---------------------------------------------------------------------------
#
# The corpus mixes three kinds of authority and they are not interchangeable:
#
#   FATF    international *standards*. Persuasive, not binding on any firm.
#   FinCEN  binding US law - the CDD rule, 31 CFR 1010.230, the CTA.
#   FFIEC   US examination procedure - what an examiner will actually test.
#
# Answering "what must a US bank collect at onboarding" out of a FATF
# Recommendation is not merely off-topic; it cites a non-binding standard as a
# legal obligation, which is the single worst failure mode a due-diligence memo
# has. Recall cannot see this - the FATF section really is about beneficial
# ownership - so routing is measured separately and boosted separately.

REGIME_OF_SOURCE = {
    "FATF": "FATF", "FATF-GUID": "FATF",
    "FinCEN": "FinCEN", "FinCEN-CTA": "FinCEN",
    "FFIEC": "FFIEC",
}

# Query-side cues. Deliberately conservative: an unmatched query routes to no
# regime and the boost simply does not apply, which is safer than guessing.
_REGIME_CUES = {
    "FinCEN": (
        r"\b31 ?cfr\b", r"\bcdd rule\b", r"\bcorporate transparency\b", r"\bcta\b",
        r"\bfincen\b", r"\breporting company\b", r"\bbeneficial ownership rule\b",
        r"\blegal entity customer\b", r"\bus (?:bank|firm|institution)\b",
    ),
    "FATF": (
        r"\bfatf\b", r"\brecommendation ?\d+\b", r"\binterpretive note\b",
        r"\binternational standard\b", r"\bmutual evaluation\b",
    ),
    "FFIEC": (
        r"\bffiec\b", r"\bexamin", r"\bmanual\b", r"\bsupervisor", r"\btesting procedure\b",
    ),
}


def infer_regime(query: str) -> str | None:
    """Which authority the question is asking about, or None if unclear."""
    low = query.lower()
    scores = {
        regime: sum(1 for pattern in patterns if re.search(pattern, low))
        for regime, patterns in _REGIME_CUES.items()
    }
    best = max(scores, key=lambda r: scores[r])
    return best if scores[best] > 0 else None


def regime_of(section: Section) -> str | None:
    return REGIME_OF_SOURCE.get(section.source)


class RegulatoryIndex:
    def __init__(self, sections: Sequence[Section] | None = None, cfg: RagConfig = DEFAULT_RAG):
        self.cfg = cfg
        self.sections = list(sections if sections is not None else load_sections())
        self.by_id = {s.id: s for s in self.sections}
        self.bm25 = BM25(self.sections)
        self.dense = DenseIndex(self.sections, cfg.embed_dim)

    def rerank(self, query: str, candidates: Sequence[str], regime: str | None = None) -> list[tuple[str, float]]:
        """Refine the fused order with query-term coverage weighted by IDF.

        Deliberately weighted below the fusion rank: on a twenty-section corpus
        the fused ranking is already close, and a reranker given the deciding
        vote makes things worse more often than better.
        """
        q_terms = set(tokenize(query))
        regime = regime if regime is not None else (infer_regime(query) if self.cfg.regime_routing else None)
        out = []
        for rank, sid in enumerate(candidates):
            section = self.by_id[sid]
            terms = set(tokenize(section.indexed_text))
            covered = sum(self.bm25.idf(t) for t in q_terms & terms)
            total = sum(self.bm25.idf(t) for t in q_terms) or 1.0
            title_hit = len(q_terms & set(tokenize(section.title))) / (len(q_terms) or 1)
            # Weighted above the fusion rank on purpose. A binding US
            # obligation and an international standard on the same subject
            # retrieve about equally well, and citing the wrong one is a legal
            # error rather than a relevance error. Off-regime sections are
            # demoted, never removed: cross-references are real, and FFIEC
            # procedure routinely explains a FinCEN rule.
            same_regime = 1.0 if (regime and regime_of(section) == regime) else 0.0
            out.append((sid, 3.0 / (1 + rank) + 1.4 * (covered / total) + 0.8 * title_hit
                        + 2.2 * same_regime))
        return sorted(out, key=lambda kv: (-kv[1], kv[0]))

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or self.cfg.top_k
        lexical = self.bm25.search(query, self.cfg.candidate_k)
        dense = self.dense.search(query, self.cfg.candidate_k)
        fused = rrf([lexical, dense], self.cfg.rrf_k)

        order = [sid for sid, _ in fused]
        if self.cfg.rerank:
            regime = infer_regime(query) if self.cfg.regime_routing else None
            order = [sid for sid, _ in self.rerank(query, order[: self.cfg.rerank_depth], regime)] + order[self.cfg.rerank_depth :]

        results = []
        for sid in order[:top_k]:
            section = self.by_id[sid]
            results.append({**section.to_dict(), "text": section.text, "regime": regime_of(section)})
        return results
