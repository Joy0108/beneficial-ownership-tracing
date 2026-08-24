from __future__ import annotations

import pytest

from ubo.er.blocking import candidate_recall, record_keys
from ubo.er.fit import TokenMemoriser, capacity_experiment
from ubo.er.normalize import (
    normalize_address,
    normalize_birth_date,
    normalize_name,
    phonetic_key,
    soundex,
    transliterate,
)
from ubo.er.resolve import UnionFind, cluster
from ubo.er.scoring import birth_date_agreement, jaro_winkler, score_pair, token_containment
from ubo.er.splits import cluster_level_split, pair_level_split
from ubo.registers.loaders import Record

# --- normalisation ---------------------------------------------------------

def test_transliteration_romanises_cyrillic_and_arabic():
    assert transliterate("Волков") == "volkov"
    assert transliterate("Кузнецова") == "kuznetsova"
    assert transliterate("Café") == "Cafe"


def test_legal_forms_are_stripped_but_distinguishing_words_are_not():
    assert normalize_name("Northwind Energy Trading Ltd") == normalize_name("NORTHWIND ENERGY TRADING LIMITED")
    # "Group" and "Enterprises" distinguish two unrelated firms; stripping them
    # would collapse the hardest negative pair in the corpus.
    assert normalize_name("Regent Ventures Group Ltd") != normalize_name("Regent Ventures Enterprises Ltd")


def test_punctuated_legal_forms_normalise_to_the_same_string():
    assert normalize_name("Emerald Resources P.L.C.") == normalize_name("Emerald Resources PLC")
    assert normalize_name("Orion Petrochemicals B.V.") == normalize_name("Orion Petrochemicals BV")


def test_a_name_is_never_stripped_to_nothing():
    assert normalize_name("Holdings Limited")


def test_soundex_survives_transliteration_variants():
    assert soundex("volkov") == soundex("wolkow"), "v/w alternation is a romanisation artefact"
    assert soundex("kuznetsova") == soundex("kouznetsova")
    assert soundex("smith") != soundex("jones")


def test_phonetic_key_ignores_word_order():
    assert phonetic_key("Dmitri Volkov") == phonetic_key("Volkov Dmitri")


def test_address_normalisation_drops_boilerplate():
    tokens = normalize_address("Suite 1, Second Floor, PO Box 71, Road Town, Tortola")
    assert "suite" not in tokens and "box" not in tokens
    assert "tortola" in tokens


def test_birth_date_precision_is_preserved():
    assert normalize_birth_date("1968-04-11") == "1968-04-11"
    assert normalize_birth_date("1968-04") == "1968-04"
    assert normalize_birth_date("") == ""


# --- similarity ------------------------------------------------------------

def test_jaro_winkler_rewards_a_shared_prefix():
    assert jaro_winkler("morozov", "morosov") > jaro_winkler("morozov", "zovmoro")


def test_containment_handles_a_truncated_name():
    assert token_containment("Baltic Resource", "Baltic Resource Holdings Group") == 1.0


def test_birth_date_conflict_is_evidence_against():
    assert birth_date_agreement("1968-04-11", "1972-04-11") < 0
    assert birth_date_agreement("1968-04", "1968-04-11") == pytest.approx(0.8)
    assert birth_date_agreement("", "1968-04-11") == 0.0


# --- scoring ---------------------------------------------------------------

def _person(rid, name, dob="", jur="RU", source="test"):
    return Record(record_id=rid, source=source, entity_type="person", name=name, birth_date=dob, jurisdiction=jur)


def test_a_shared_identifier_settles_a_match():
    a = Record("a", "gleif_l1", "company", "Northwind Energy", identifier="5493001KJTIIGC8Y1R12")
    b = Record("b", "aggregator", "company", "Nrthwind Enrgy Ltd", identifier="5493001KJTIIGC8Y1R12")
    assert score_pair(a, b).decision == "match"


def test_a_birth_date_conflict_blocks_a_match_however_good_the_name():
    a = _person("a", "Dmitri Volkov", "1968-04-11")
    b = _person("b", "Dmitri Volkov", "1975-01-02")
    assert score_pair(a, b).decision == "reject"


def test_a_person_and_a_company_are_never_the_same_entity():
    a = _person("a", "Meridian Nominees")
    b = Record("b", "gleif_l1", "company", "Meridian Nominees")
    assert score_pair(a, b).decision == "reject"


def test_aliases_are_scored_not_just_the_primary_name():
    a = Record("a", "opensanctions", "person", "Дмитрий Волков", aliases=("Dmitry Volkov",))
    b = _person("b", "Dmitri Volkov", source="ofac")
    assert score_pair(a, b).features["name_jaro"] > 0.9


# --- blocking --------------------------------------------------------------

def test_blocking_keys_are_namespaced_by_entity_type():
    person = _person("a", "Meridian Nominees")
    company = Record("b", "gleif_l1", "company", "Meridian Nominees")
    assert not set(record_keys(person)) & set(record_keys(company))


def test_aliases_produce_blocking_keys():
    with_alias = Record("a", "opensanctions", "person", "Дмитрий Волков", aliases=("Dmitry Volkov",))
    plain = _person("b", "Dmitry Volkov")
    assert set(record_keys(with_alias)) & set(record_keys(plain))


def test_blocking_reduces_the_pair_space_without_losing_matches(candidates, truth):
    pairs, report = candidates
    assert report.reduction_ratio > 0.95, f"only {report.reduction_ratio:.3f} of pairs eliminated"
    recall = candidate_recall(pairs, truth)["candidate_recall"]
    assert recall >= 0.97, f"blocking lost {1 - recall:.1%} of true pairs, which nothing downstream can recover"


def test_shared_service_addresses_do_not_dominate_candidate_generation(candidates):
    _pairs, report = candidates
    by_type = report.by_key_type
    address_share = by_type.get("address", 0) / max(1, report.n_candidate_pairs)
    assert address_share < 0.3, "address blocking is generating most of the candidates, which means it is not filtered"


# --- clustering ------------------------------------------------------------

def test_union_find_merges_transitively():
    uf = UnionFind(["a", "b", "c", "d"])
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("c")
    assert uf.find("d") != uf.find("a")


def test_cluster_records_the_weakest_link_that_holds_it_together():
    records = [Record(str(i), "test", "person", f"Person {i}") for i in range(3)]
    entities = cluster(records, {("0", "1"), ("1", "2")}, {("0", "1"): 0.99, ("1", "2"): 0.61})
    merged = next(e for e in entities if len(e.record_ids) == 3)
    assert merged.weakest_link == pytest.approx(0.61)


def test_adjudication_only_moves_the_borderline_band(scores, resolution):
    accepted = resolution["accepted"]
    clear_matches = {(s.left, s.right) for s in scores if s.decision == "match"}
    rejects = {(s.left, s.right) for s in scores if s.decision == "reject"}
    assert clear_matches <= accepted, "a clear match must not be dropped by adjudication"
    assert not (rejects & accepted), "a clear reject must not be promoted by adjudication"


def test_resolution_beats_a_name_similarity_baseline(by_id, candidates, truth, resolution):
    from dataclasses import replace

    from ubo.config import DEFAULT_SCORING
    from ubo.er.scoring import score_candidates

    baseline_cfg = replace(DEFAULT_SCORING, weights={"name_jaro": 1.0},
                           match_threshold=0.85, review_low=0.85, review_high=0.85)
    baseline = {(s.left, s.right) for s in score_candidates(by_id, candidates[0], baseline_cfg) if s.decision == "match"}
    predicted = resolution["accepted"]

    def f1(pred):
        tp, fp, fn = len(pred & truth), len(pred - truth), len(truth - pred)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    assert f1(predicted) > f1(baseline) + 0.2


# --- splits and leakage ----------------------------------------------------

def test_pair_level_split_leaks_records_and_cluster_level_does_not(clusters):
    assert pair_level_split(clusters).shared_records, "the naive split is supposed to leak; that is the point"
    assert not cluster_level_split(clusters).shared_records


def test_cluster_split_keeps_every_entity_whole(clusters):
    split = cluster_level_split(clusters)
    for members in clusters.values():
        members = set(members)
        assert not (members & split.train_records and members & split.test_records)


def test_a_token_level_model_is_inflated_by_the_pair_level_split(by_id, clusters, candidates):
    result = capacity_experiment(by_id, clusters, candidates[0])
    assert result["verdict"]["inflation"] > 0.1, (
        "a model with memorisation capacity must score higher under the leaking split; "
        "if it does not, the experiment is not measuring leakage"
    )


def test_the_memoriser_needs_negatives_to_be_a_fair_comparison(by_id, clusters, candidates):
    from ubo.er.blocking import true_pairs as tp_fn
    from ubo.er.splits import cluster_level_split, negatives_for

    split = cluster_level_split(clusters)
    positives = tp_fn(clusters)
    negatives = negatives_for(split, candidates[0], positives, "train")
    without = TokenMemoriser(by_id).train(split.train_pairs)
    with_negatives = TokenMemoriser(by_id).train(split.train_pairs, negatives)
    assert len(with_negatives.matched_tokens) < len(without.matched_tokens)
