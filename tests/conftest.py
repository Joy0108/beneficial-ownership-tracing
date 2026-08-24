from __future__ import annotations

import pytest

from ubo.er.adjudicate import adjudicate_all
from ubo.er.blocking import candidate_pairs, true_pairs
from ubo.er.resolve import accepted_pairs, cluster, record_to_entity
from ubo.er.scoring import score_candidates
from ubo.eval.truth import layering_labels, load_world, map_entities_to_seed, truth_clusters
from ubo.graph.build import build_graph
from ubo.graph.features import compute_features
from ubo.rag.retrieve import RegulatoryIndex
from ubo.registers.loaders import load_all


@pytest.fixture(scope="session")
def loaded():
    records, statements = load_all()
    if not records:
        pytest.skip("registers not built; run `python scripts/build_registers.py`")
    return records, statements


@pytest.fixture(scope="session")
def records(loaded):
    return loaded[0]


@pytest.fixture(scope="session")
def statements(loaded):
    return loaded[1]


@pytest.fixture(scope="session")
def by_id(records):
    return {r.record_id: r for r in records}


@pytest.fixture(scope="session")
def world():
    return load_world()


@pytest.fixture(scope="session")
def clusters(world):
    return truth_clusters(world)


@pytest.fixture(scope="session")
def truth(clusters):
    return true_pairs(clusters)


@pytest.fixture(scope="session")
def candidates(records):
    pairs, report = candidate_pairs(records)
    return pairs, report


@pytest.fixture(scope="session")
def scores(by_id, candidates):
    return score_candidates(by_id, candidates[0])


@pytest.fixture(scope="session")
def resolution(records, by_id, scores):
    adjudications, stats = adjudicate_all(by_id, scores)
    accepted, strength = accepted_pairs(scores, adjudications)
    entities = cluster(records, accepted, strength)
    return {"accepted": accepted, "entities": entities, "adjudication": stats}


@pytest.fixture(scope="session")
def entities(resolution):
    return resolution["entities"]


@pytest.fixture(scope="session")
def graph(entities, statements):
    return build_graph(entities, statements, record_to_entity(entities))


@pytest.fixture(scope="session")
def mapping(entities, world):
    return map_entities_to_seed(entities, world)[0]


@pytest.fixture(scope="session")
def labels(world, mapping):
    return layering_labels(world, mapping)


@pytest.fixture(scope="session")
def features(graph, labels):
    return compute_features(graph, roots=list(labels))


@pytest.fixture(scope="session")
def regulatory():
    return RegulatoryIndex()
