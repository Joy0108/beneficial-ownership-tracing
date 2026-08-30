"""Command line entry point: ``ubo <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from .config import REPORT_DIR


def _pipeline():
    """Resolve once and hand the pieces back. Shared by several commands."""
    from .er.adjudicate import adjudicate_all
    from .er.blocking import candidate_pairs
    from .er.resolve import accepted_pairs, cluster, record_to_entity
    from .er.scoring import score_candidates
    from .graph.build import build_graph
    from .registers.loaders import load_all

    records, statements = load_all()
    by_id = {r.record_id: r for r in records}
    candidates, blocking = candidate_pairs(records)
    scores = score_candidates(by_id, candidates)
    adjudications, adj_stats = adjudicate_all(by_id, scores)
    accepted, strength = accepted_pairs(scores, adjudications)
    entities = cluster(records, accepted, strength)
    graph = build_graph(entities, statements, record_to_entity(entities))
    return records, statements, entities, graph, blocking, adj_stats


def cmd_registers(args) -> int:
    from .registers.loaders import load_all, source_counts

    records, statements = load_all()
    from collections import Counter

    print(json.dumps({
        "records": len(records),
        "by_source": source_counts(records),
        "by_type": dict(Counter(r.entity_type for r in records)),
        "statements": len(statements),
        "statements_by_source": dict(Counter(s.source for s in statements)),
    }, indent=2))
    return 0


def cmd_resolve(args) -> int:
    from .er.resolve import cluster_summary

    _records, _statements, entities, _graph, blocking, adj = _pipeline()
    print(json.dumps({"blocking": blocking.to_dict(), "adjudication": adj,
                      "clusters": cluster_summary(entities)}, indent=2))
    if args.show:
        for entity in sorted(entities, key=lambda e: -len(e.record_ids))[: args.show]:
            print(f"\n{entity.entity_id}  {entity.canonical_name}  ({entity.entity_type}, weakest link {entity.weakest_link:.2f})")
            for prov in entity.provenance:
                print(f"    {prov['source']:<16} {prov['name_as_recorded']}")
    return 0


def cmd_graph(args) -> int:
    from .graph.features import compute_features, describe
    from .graph.patterns import assess

    _records, _statements, _entities, graph, _b, _a = _pipeline()
    print(json.dumps(graph.stats(), indent=2))

    roots = [e for e in graph.entities if graph.children(e, control_only=True) and not graph.parents(e, control_only=True)]
    features = compute_features(graph, roots=roots)
    rows = sorted(((assess(f), f) for f in features.values()), key=lambda p: -p[0].score)
    print("\nstructures by risk score:")
    for assessment, feats in rows[: args.top]:
        entity = graph.entities[assessment.entity_id]
        print(f"\n  {entity.canonical_name}  score {assessment.score:.2f}  {'FLAGGED' if assessment.flagged else 'below threshold'}")
        for reason in describe(feats):
            print(f"    - {reason}")
    return 0


def cmd_screen(args) -> int:
    from .workflow.nodes import record_decision, screen

    records, statements, entities, graph, _b, _a = _pipeline()
    subject = args.entity
    if subject not in graph.entities:
        matches = [e.entity_id for e in entities if args.entity.lower() in e.canonical_name.lower()]
        if not matches:
            print(f"no entity matching {args.entity!r}", file=sys.stderr)
            return 1
        subject = matches[0]
        print(f"resolved {args.entity!r} to {subject} ({graph.entities[subject].canonical_name})\n", file=sys.stderr)

    state = screen(subject, state={"records": records, "statements": statements, "entities": entities})
    print(state["memo"].text)
    print("\n---\nverification:", json.dumps(state["verification"], indent=2))
    print("workflow path:", " -> ".join(state["_path"]))

    if args.decide:
        state = record_decision(state, args.decide, args.analyst or "cli-user",
                                datetime.now(timezone.utc).isoformat(timespec="seconds"))
        print("decision recorded:", json.dumps(state["decision_package"], indent=2, default=str))
    else:
        print("\nAwaiting human decision. Re-run with --decide escalate|clear|request_more_information")
    return 0


def cmd_guidance(args) -> int:
    from .rag.retrieve import RegulatoryIndex

    for hit in RegulatoryIndex().search(args.question, args.k):
        print(f"\n[{hit['id']}] {hit['title']}  ({hit['source']})")
        print(f"  {hit['text'][:300]}...")
    return 0


def cmd_eval(args) -> int:
    from .eval.run_eval import metrics_for_registry, run_full_eval
    from .registry import Registry, gate_report

    report = run_full_eval(deep=not args.fast)
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2, default=str))

    metrics = metrics_for_registry(report)
    registry = Registry()
    version = registry.register(metrics, params={"blocking": report["config"]["blocking"]},
                                notes=args.note or "make eval")
    print("\n" + gate_report(version))

    if not version.gates_passed and not args.no_gate:
        print("\nPROMOTION GATES FAILED", file=sys.stderr)
        return 1
    print(f"\nreport written to {REPORT_DIR / 'eval_report.json'}")
    return 0


def cmd_registry(args) -> int:
    from .registry import Registry, gate_report

    registry = Registry()
    if args.promote:
        result = registry.promote(args.promote, force=args.force)
        print(json.dumps(result, indent=2))
        return 0 if result["promoted"] else 1
    print(json.dumps(registry.summary(), indent=2))
    for version in registry.versions:
        print()
        print(gate_report(version))
    return 0


def cmd_drift(args) -> int:
    from .drift import run_monitors, simulate_designation_round
    from .registers.loaders import load_all

    records, _ = load_all()
    current = simulate_designation_round(records, args.jurisdiction, args.multiplier) if args.simulate else records
    print(json.dumps(run_monitors(records, current), indent=2))
    return 0


def cmd_workflow(args) -> int:
    from .workflow.nodes import build_workflow

    wf = build_workflow(engine=getattr(args, "engine", "auto"))
    print(f"%% engine: {wf.engine}")
    print(wf.to_mermaid())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ubo", description="Beneficial ownership tracing and sanctions evasion detection")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("registers", help="summarise the loaded registers")
    p.set_defaults(func=cmd_registers)

    p = sub.add_parser("resolve", help="run entity resolution and report")
    p.add_argument("--show", type=int, default=0, help="print the N largest clusters with provenance")
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("graph", help="build the ownership graph and rank structures by risk")
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--engine", choices=["auto", "langgraph", "reference"], default="auto",
                   help="execution engine; auto picks langgraph when installed")
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("screen", help="run the eight-step workflow on one entity")
    p.add_argument("entity", help="entity id, or a substring of the canonical name")
    p.add_argument("--decide", choices=["escalate", "clear", "request_more_information"])
    p.add_argument("--analyst", help="analyst identity recorded against the decision")
    p.add_argument("--engine", choices=["auto", "langgraph", "reference"], default="auto",
                   help="execution engine; auto picks langgraph when installed")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("guidance", help="query the regulatory corpus")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_guidance)

    p = sub.add_parser("eval", help="run the full evaluation and register the result")
    p.add_argument("--fast", action="store_true", help="skip leave-one-out and the split experiments")
    p.add_argument("--no-gate", action="store_true", help="report without failing on a gate breach")
    p.add_argument("--note")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("registry", help="show registered versions, or promote one")
    p.add_argument("--promote", type=int, metavar="VERSION")
    p.add_argument("--force", action="store_true", help="promote despite failed blocking gates")
    p.set_defaults(func=cmd_registry)

    p = sub.add_parser("drift", help="run the drift monitors")
    p.add_argument("--simulate", action="store_true", help="simulate a sanctions designation round")
    p.add_argument("--jurisdiction", default="RU")
    p.add_argument("--multiplier", type=int, default=4)
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("workflow", help="print the screening workflow as mermaid")
    p.set_defaults(func=cmd_workflow)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
