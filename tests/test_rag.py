from __future__ import annotations

import pytest

from ubo.eval.rag import evaluate_pep_language, evaluate_retrieval, load_golden
from ubo.rag.memo import CITATION, Memo, verify_citations
from ubo.rag.retrieve import rrf, tokenize


def test_spelled_out_numbers_and_numerals_share_tokens():
    """Regulatory text writes "twenty-five percent"; queries write "25%"."""
    assert "25" in tokenize("twenty-five percent threshold")
    assert "twenty" in tokenize("25 percent threshold")


def test_section_identifiers_are_searchable(regulatory):
    hits = [h["id"] for h in regulatory.search("31 CFR 1010.230 beneficial ownership legal entity customers", 3)]
    assert "FinCEN-CDD-1" in hits


def test_rrf_rewards_agreement_between_retrievers():
    lexical = [("a", 9.0), ("b", 8.0)]
    dense = [("b", 0.9), ("c", 0.8)]
    fused = dict(rrf([lexical, dense], k=10))
    assert fused["b"] > fused["a"] and fused["b"] > fused["c"]


def test_retrieval_finds_a_primary_section_for_every_golden_question(regulatory):
    result = evaluate_retrieval(regulatory, load_golden(), k=5)
    assert result["any_primary@5"] == 1.0, [r for r in result["rows"] if not r["any_primary@k"]]
    assert result["recall@5"] >= 0.9
    assert result["mrr"] >= 0.85


def test_the_pep_language_gate_is_perfect_on_its_probes(regulatory):
    """This one is a correctness requirement, not a quality metric."""
    result = evaluate_pep_language(regulatory, load_golden(), k=5)
    assert result["language_rate"] == 1.0, result["probe_detail"]
    assert result["retrieval_rate"] == 1.0


def test_the_indirect_ownership_question_retrieves_the_multiplication_rule(regulatory):
    hits = [h["id"] for h in regulatory.search("how is indirect ownership through intermediate entities calculated", 5)]
    assert "FinCEN-CDD-2" in hits


def test_the_fifty_percent_rule_is_retrievable_by_description(regulatory):
    hits = [h["id"] for h in regulatory.search(
        "entity not on the sanctions list but owned by blocked persons in aggregate", 5)]
    assert "FFIEC-SANCTIONS" in hits


def test_citation_syntax_does_not_match_ordinary_brackets():
    assert CITATION.findall("The holding [see figure 2] is 25% [reg:FATF-R10].") == ["FATF-R10"]


def test_verify_citations_flags_a_reference_to_nothing():
    memo = Memo("E-1", "The threshold is twenty-five percent [reg:MADE-UP-SECTION].")
    result = verify_citations(memo, valid_statements=set(), valid_sections={"FATF-R10"})
    assert result["unresolved"] == ["MADE-UP-SECTION"]
    assert result["resolution_rate"] == 0.0


def test_verify_citations_accepts_statement_and_section_references():
    memo = Memo("E-1", "Alpha holds 60% of Beta [stmt:psc-000001]. The threshold is 25% [reg:FATF-R10].")
    result = verify_citations(memo, valid_statements={"psc-000001"}, valid_sections={"FATF-R10"})
    assert result["resolution_rate"] == 1.0


@pytest.mark.parametrize("query,expected", [
    ("who must a trustee identify for an express trust", "FATF-R25"),
    ("nominee shareholder acting on undisclosed instructions", "FATF-LAYERING"),
    ("is exiting a whole category of customer acceptable", "FATF-RISK-BASED"),
    ("how long must records be kept", "FATF-RECORDS"),
])
def test_paraphrased_questions_reach_the_right_section(regulatory, query, expected):
    assert expected in [h["id"] for h in regulatory.search(query, 5)]
