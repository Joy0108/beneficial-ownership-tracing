"""Model registry with promotion gates.

A local, file-backed stand-in for MLflow's model registry, with the part that
actually matters kept: a version cannot reach production because someone decided
it was better. It reaches production by clearing named gates, and the gate
results are stored with the version so the promotion can be re-examined later.

Two gates here are not accuracy thresholds and are the reason this is not just a
leaderboard:

* ``pep_language`` is absolute. A version that renders a PEP match as an adverse
  finding does not ship at any accuracy.
* ``candidate_recall`` guards a ceiling rather than a score. Blocking recall
  bounds everything downstream, and a version that improved F1 by throwing away
  candidates has not improved.

``mlflow`` is used instead when installed and ``UBO_MLFLOW_URI`` is set; the
gate logic is identical and lives here either way.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ARTIFACT_DIR

Stage = str  # "none" | "staging" | "production" | "archived"


@dataclass
class Gate:
    name: str
    metric: str
    minimum: float
    blocking: bool = True
    rationale: str = ""

    def evaluate(self, metrics: dict[str, float]) -> dict[str, Any]:
        value = metrics.get(self.metric)
        passed = value is not None and value >= self.minimum
        return {
            "gate": self.name,
            "metric": self.metric,
            "minimum": self.minimum,
            "actual": value,
            "passed": bool(passed),
            "blocking": self.blocking,
            "rationale": self.rationale,
        }


DEFAULT_GATES = [
    Gate("entity_resolution_f1", "er_f1", 0.90, True,
         "Below this, false merges attribute one party's holdings to another and the graph is wrong in a way no "
         "downstream stage can detect."),
    Gate("candidate_recall", "candidate_recall", 0.97, True,
         "Blocking recall is a hard ceiling on the pipeline. A version that raised F1 by generating fewer candidates "
         "has not improved, it has narrowed."),
    Gate("er_precision", "er_precision", 0.95, True,
         "A false merge is silent and unrecoverable. Precision is gated above recall for that reason."),
    Gate("layering_precision", "layering_precision", 0.80, True,
         "Every false flag is an analyst-day. Below this the queue stops being worked."),
    Gate("regulatory_recall_at_5", "rag_recall_at_5", 0.85, True,
         "A memo that misses the governing section is not a citation problem, it is a wrong memo."),
    Gate("citation_resolution", "citation_resolution", 0.99, True,
         "A citation that resolves to nothing is a fabricated reference in a regulated document."),
    Gate("pep_language", "pep_language_rate", 1.0, True,
         "Absolute. Rendering a PEP match as an adverse finding is wrong on the law and is the mechanism of "
         "unjustified de-risking. No accuracy score compensates."),
]


@dataclass
class Version:
    version: int
    created_at: str
    stage: Stage = "none"
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    gate_results: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    @property
    def gates_passed(self) -> bool:
        return all(g["passed"] for g in self.gate_results if g["blocking"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Registry:
    def __init__(self, name: str = "ubo-screening", path: Path | None = None, gates: list[Gate] | None = None):
        self.name = name
        self.path = path or (ARTIFACT_DIR / "registry.json")
        self.gates = gates if gates is not None else list(DEFAULT_GATES)
        self.versions: list[Version] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.versions = [Version(**v) for v in payload.get("versions", [])]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"model": self.name, "versions": [v.to_dict() for v in self.versions]}, indent=2),
            encoding="utf-8", newline="\n",
        )

    def register(self, metrics: dict[str, float], params: dict[str, Any] | None = None, notes: str = "") -> Version:
        version = Version(
            version=len(self.versions) + 1,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            metrics=dict(metrics),
            params=dict(params or {}),
            gate_results=[g.evaluate(metrics) for g in self.gates],
            notes=notes,
        )
        self.versions.append(version)
        self._save()
        _mlflow_log(self.name, version)
        return version

    def promote(self, version: int, to: Stage = "production", force: bool = False) -> dict[str, Any]:
        target = next((v for v in self.versions if v.version == version), None)
        if target is None:
            raise KeyError(f"no version {version}")

        failures = [g for g in target.gate_results if g["blocking"] and not g["passed"]]
        if failures and not force:
            return {
                "promoted": False,
                "version": version,
                "to": to,
                "failed_gates": failures,
                "reason": "blocking gates failed; promotion refused",
            }

        for other in self.versions:
            if other.stage == to and other.version != version:
                other.stage = "archived"
        target.stage = to
        self._save()
        return {
            "promoted": True,
            "version": version,
            "to": to,
            "forced": bool(failures and force),
            "failed_gates": failures,
        }

    def production(self) -> Version | None:
        return next((v for v in self.versions if v.stage == "production"), None)

    def compare_to_production(self, metrics: dict[str, float]) -> dict[str, Any]:
        current = self.production()
        if current is None:
            return {"baseline": None, "deltas": {}}
        deltas = {k: round(v - current.metrics.get(k, 0.0), 4) for k, v in metrics.items() if k in current.metrics}
        regressions = {k: d for k, d in deltas.items() if d < -0.01}
        return {
            "baseline_version": current.version,
            "deltas": deltas,
            "regressions": regressions,
            "clean": not regressions,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "versions": len(self.versions),
            "production": self.production().version if self.production() else None,
            "history": [
                {"version": v.version, "stage": v.stage, "gates_passed": v.gates_passed,
                 "er_f1": v.metrics.get("er_f1"), "created_at": v.created_at}
                for v in self.versions
            ],
        }


def _mlflow_log(model: str, version: Version) -> None:  # pragma: no cover - optional dependency
    uri = os.environ.get("UBO_MLFLOW_URI")
    if not uri:
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        with mlflow.start_run(run_name=f"{model}-v{version.version}"):
            mlflow.log_params({k: str(v) for k, v in version.params.items()})
            mlflow.log_metrics(version.metrics)
            mlflow.set_tag("gates_passed", version.gates_passed)
    except Exception:
        # Telemetry must never take the pipeline down with it.
        return


def gate_report(version: Version) -> str:
    lines = [f"version {version.version} ({version.stage})", ""]
    for gate in version.gate_results:
        mark = "PASS" if gate["passed"] else "FAIL"
        actual = "n/a" if gate["actual"] is None else f"{gate['actual']:.4f}"
        lines.append(f"  [{mark}] {gate['gate']:<24} {actual} >= {gate['minimum']}")
        if not gate["passed"] and gate["rationale"]:
            lines.append(f"         {gate['rationale']}")
    return "\n".join(lines)


def default_gate_callable() -> Callable[[dict[str, float]], list[dict[str, Any]]]:
    return lambda metrics: [g.evaluate(metrics) for g in DEFAULT_GATES]
