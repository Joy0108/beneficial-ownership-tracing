"""Configuration. Everything an experiment varies lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("UBO_DATA_DIR", ROOT / "data"))
REGISTER_DIR = DATA_DIR / "registers"
WORLD_PATH = DATA_DIR / "world" / "seed_world.json"
REGULATORY_DIR = DATA_DIR / "regulatory"
GOLDEN_PATH = DATA_DIR / "golden" / "rag_golden.json"
ARTIFACT_DIR = Path(os.environ.get("UBO_ARTIFACT_DIR", ROOT / "artifacts"))
REPORT_DIR = Path(os.environ.get("UBO_REPORT_DIR", ROOT / "reports"))

# Jurisdictions the FATF and the Tax Justice Network's Financial Secrecy Index
# both flag for opacity of beneficial ownership. Used as a structural feature,
# never on its own as a risk verdict.
SECRECY_JURISDICTIONS = frozenset({"KY", "VG", "PA", "SC", "BZ", "AE", "CY", "MT", "BS", "BM", "JE", "GG", "IM", "LI"})


@dataclass(frozen=True)
class BlockingConfig:
    """Candidate generation.

    Recall here is a hard ceiling on the whole pipeline: a true pair that never
    becomes a candidate can never be recovered by any scorer downstream. The
    keys are therefore redundant on purpose - each one fails on a different
    defect, and a pair only has to survive one of them.
    """

    name: str = "default"
    use_name_tokens: bool = True     # sorted significant tokens; survives word order
    use_prefix: bool = True          # normalised name prefix; survives suffix noise
    use_phonetic: bool = True        # soundex-like code; survives transliteration
    use_address: bool = True         # address token; survives name changes entirely
    use_identifier: bool = True      # LEI / company number; exact, near-free
    prefix_length: int = 6
    token_key_size: int = 2          # how many sorted tokens make a key
    max_block_size: int = 60         # oversized blocks are dropped, and reported
    # An address token shared by more than this many records belongs to a
    # corporate service provider, not to an entity. Blocking on it generates
    # thousands of candidate pairs and finds nothing. The shared-address signal
    # is kept for the graph stage, where it does mean something.
    max_address_df: int = 8

    def variant(self, name: str, **overrides) -> BlockingConfig:
        return replace(self, name=name, **overrides)


@dataclass(frozen=True)
class ScoringConfig:
    name: str = "default"
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "name_jaro": 0.34,
            "name_token_jaccard": 0.22,
            "phonetic_match": 0.08,
            "address_overlap": 0.12,
            "jurisdiction_match": 0.08,
            "birth_date_match": 0.12,
            "identifier_match": 0.04,
        }
    )
    match_threshold: float = 0.72
    review_low: float = 0.58   # below this, reject outright
    review_high: float = 0.80  # above this, accept without adjudication
    type_must_match: bool = True

    def variant(self, name: str, **overrides) -> ScoringConfig:
        return replace(self, name=name, **overrides)


@dataclass(frozen=True)
class GraphConfig:
    control_threshold: float = 25.0     # percent, the common BO disclosure floor
    max_chain_depth: int = 12
    attenuate_ownership: bool = True    # multiply percentages along a chain


@dataclass(frozen=True)
class RagConfig:
    top_k: int = 5
    candidate_k: int = 25
    rrf_k: int = 60
    rerank: bool = True
    # Boost sections from the authority the question is actually about.
    # FATF is a standard; FinCEN is law. Citing one for the other is the worst
    # failure a due-diligence memo has, and recall cannot see it.
    regime_routing: bool = True
    rerank_depth: int = 15
    embed_dim: int = 96
    require_citations: bool = True


@dataclass(frozen=True)
class AdjudicationConfig:
    backend: str = os.environ.get("UBO_LLM", "deterministic")  # or "anthropic"
    anthropic_model: str = "claude-opus-5"
    max_adjudications: int = 400


DEFAULT_BLOCKING = BlockingConfig()
DEFAULT_SCORING = ScoringConfig()
DEFAULT_GRAPH = GraphConfig()
DEFAULT_RAG = RagConfig()
DEFAULT_ADJUDICATION = AdjudicationConfig()


def ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
