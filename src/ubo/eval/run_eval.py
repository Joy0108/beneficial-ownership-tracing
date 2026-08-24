"""The end-to-end evaluation. ``make eval`` runs this and CI gates on it.

Every stage is measured separately, because they fail separately and a single
composite number would hide which one broke. The layering score in particular
depends on entity resolution having worked: a structure split across two
resolved entities loses its chain, and that shows up as a graph miss unless the
two are reported side by side.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import (
    DEFAULT_ADJUDICATION,
    DEFAULT_BLOCKING,
    DEFAULT_GRAPH,
    DEFAULT_RAG,
    DEFAULT_SCORING,
    REPORT_DIR,
    ensure_dirs,
)
from ..er.adjudicate import adjudicate_all
from ..er.blocking import candidate_pairs, candidate_recall, true_pairs
from ..er.fit import capacity_experiment, split_experiment
from ..er.resolve import accepted_pairs, cluster, cluster_summary, record_to_entity
from ..er.scoring import score_candidates
from ..graph.build import build_graph
from ..graph.features import compute_features
from ..graph.patterns import DEFAULT_THRESHOLD, assess_all, sweep_threshold
from ..graph.patterns import evaluate as evaluate_layering
from ..rag.retrieve import RegulatoryIndex
from ..registers.loaders import load_all, source_counts
from ..workflow.nodes import screen
from .models import compare as compare_models
from .rag import ablate as rag_ablate
from .rag import evaluate_pep_language, evaluate_retrieval, load_golden
from .truth import layering_labels, load_world, map_entities_to_seed, truth_clusters


def _prf(predicted: set, truth: set) -> dict[str, float]:
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def run_full_eval(out_dir: Path = REPORT_DIR, write: bool = True, deep: bool = True) -> dict[str, Any]:
    ensure_dirs()
    records, statements = load_all()
    by_id = {r.record_id: r for r in records}
    world = load_world()
    clusters = truth_clusters(world)
    truth = true_pairs(clusters)

    # --- entity resolution -------------------------------------------------
    candidates, blocking_report = candidate_pairs(records, DEFAULT_BLOCKING)
    recall_report = candidate_recall(candidates, truth)
    scores = score_candidates(by_id, candidates, DEFAULT_SCORING)
    adjudications, adj_stats = adjudicate_all(by_id, scores, cfg=DEFAULT_ADJUDICATION)
    accepted, strength = accepted_pairs(scores, adjudications)
    er = _prf(accepted, truth)

    # Baseline: name similarity over the same candidates, no normalisation
    # beyond lowercasing, no adjudication. What the pipeline is worth is the
    # distance from here, not the absolute number.
    from dataclasses import replace

    baseline_cfg = replace(
        DEFAULT_SCORING, name="name-similarity-only",
        weights={"name_jaro": 1.0}, match_threshold=0.85, review_low=0.85, review_high=0.85,
    )
    baseline_scores = score_candidates(by_id, candidates, baseline_cfg)
    baseline = _prf({(s.left, s.right) for s in baseline_scores if s.decision == "match"}, truth)

    entities = cluster(records, accepted, strength)
    mapping, mapping_stats = map_entities_to_seed(entities, world)

    # --- graph -------------------------------------------------------------
    graph = build_graph(entities, statements, record_to_entity(entities), DEFAULT_GRAPH)
    labels = layering_labels(world, mapping)
    features = compute_features(graph, DEFAULT_GRAPH, roots=list(labels))
    assessments = assess_all(features, threshold=DEFAULT_THRESHOLD)
    layering = evaluate_layering(assessments, labels)

    # --- regulatory retrieval ---------------------------------------------
    golden = load_golden()
    index = RegulatoryIndex()
    rag = evaluate_retrieval(index, golden, DEFAULT_RAG.top_k)
    pep = evaluate_pep_language(index, golden, DEFAULT_RAG.top_k)

    # --- one end-to-end screening run -------------------------------------
    subject = _pick_subject(mapping, entities, graph)
    workflow_state = screen(
        subject, state={"records": records, "statements": statements, "entities": entities}, regulatory=index
    )
    verification = workflow_state["verification"]

    report: dict[str, Any] = {
        "corpus": {
            "records": len(records),
            "statements": len(statements),
            "by_source": source_counts(records),
            "true_entities": len(world["truth_clusters"]),
            "resolvable_entities": len(clusters),
            "true_pairs": len(truth),
        },
        "entity_resolution": {
            "blocking": blocking_report.to_dict(),
            "candidate_recall": recall_report,
            "adjudication": adj_stats,
            "baseline_name_similarity_only": baseline,
            "pipeline": er,
            "improvement_f1": round(er["f1"] - baseline["f1"], 4),
            "clusters": cluster_summary(entities),
            "mapping_to_truth": {k: v for k, v in mapping_stats.items() if k != "false_merge_detail"},
        },
        "graph": {
            "stats": graph.stats(),
            "labelled_structures": len(labels),
            "layering": {k: v for k, v in layering.items() if k != "flagged"},
            "threshold": DEFAULT_THRESHOLD,
            "threshold_sweep": sweep_threshold(assessments, labels)[::4],
        },
        "regulatory_rag": {
            **{k: v for k, v in rag.items() if k != "rows"},
            "pep_language": {k: v for k, v in pep.items() if k not in {"rows", "probe_detail"}},
            "ablation": rag_ablate(golden, DEFAULT_RAG.top_k),
        },
        "workflow": {
            "subject": subject,
            "path": workflow_state["_path"],
            "citation_resolution": verification["citations"]["resolution_rate"],
            "citations": verification["citations"]["citations"],
            "pep_check_passed": verification["pep"]["passed"],
            "awaiting_human_decision": workflow_state["awaiting_human_decision"],
            "recommendation": workflow_state["decision_package"]["recommendation"],
        },
        "config": {
            "blocking": asdict(DEFAULT_BLOCKING),
            "scoring": asdict(DEFAULT_SCORING),
            "graph": asdict(DEFAULT_GRAPH),
            "rag": asdict(DEFAULT_RAG),
        },
    }

    if deep:
        report["leakage"] = {
            "linear_scorer": split_experiment(clusters, scores, candidates, trials=200)["verdict"],
            "token_level_model": capacity_experiment(by_id, clusters, candidates)["verdict"],
        }
        report["model_comparison"] = {
            k: v for k, v in compare_models(graph, features, labels).items() if k != "gcn"
        } | {"gcn": {kk: {k2: v2 for k2, v2 in vv.items() if k2 != "per_fold"}
                     for kk, vv in compare_models(graph, features, labels)["gcn"].items()}}

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8", newline="\n")
    return report


def _pick_subject(mapping: dict[str, str], entities, graph) -> str:
    """A resolved entity that actually holds something, for the end-to-end run."""
    with_edges = [e for e in graph.entities if graph.children(e, control_only=True)]
    for entity_id in with_edges:
        if mapping.get(entity_id, "").startswith("P"):
            return entity_id
    return with_edges[0] if with_edges else next(iter(graph.entities))


def metrics_for_registry(report: dict[str, Any]) -> dict[str, float]:
    """Flatten the report into the metric names the promotion gates reference."""
    er = report["entity_resolution"]
    return {
        "er_f1": er["pipeline"]["f1"],
        "er_precision": er["pipeline"]["precision"],
        "er_recall": er["pipeline"]["recall"],
        "candidate_recall": er["candidate_recall"]["candidate_recall"],
        "blocking_reduction": er["blocking"]["reduction_ratio"],
        "layering_precision": report["graph"]["layering"].get("precision", 0.0),
        "layering_recall": report["graph"]["layering"].get("recall", 0.0),
        "rag_recall_at_5": report["regulatory_rag"].get("recall@5", 0.0),
        "rag_mrr": report["regulatory_rag"].get("mrr", 0.0),
        "citation_resolution": report["workflow"]["citation_resolution"],
        "pep_language_rate": report["regulatory_rag"]["pep_language"]["language_rate"],
    }
