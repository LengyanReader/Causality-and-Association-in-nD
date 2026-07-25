"""UCL artifact contracts (plan §2).

Every station of the Universal Causal Loop consumes and produces one of these
typed artifacts. They are the *only* objects allowed to flow between stations —
this is what makes the loop closeable, auditable, and self-evolving.

Design rules:
- Every artifact carries `graph_version` (commit hash of the AssumptionGraph it
  was derived under) so loop invariant 1 (estimate ↔ identification ↔ graph
  version) is checkable everywhere.
- Everything serializes to plain dict/JSON for provenance logging.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any


def graph_hash(edges: list[tuple[str, str]]) -> str:
    """Deterministic version hash for an edge set (the 'commit hash' of a graph)."""
    canon = "\n".join(f"{a}->{b}" for a, b in sorted(edges))
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


@dataclass
class EstimandSpec:
    """Station 0 (FRAME): the causal question, target-trial style."""

    name: str
    question: str
    treatment: str
    outcome: str
    estimand: str  # "ATE" | "ATT" | "CATE" | "LATE"
    rung: int  # Pearl's ladder: 1 association, 2 intervention, 3 counterfactual
    population: str
    decision_context: str
    subgroup: str | None = None  # for CATE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AssumptionGraph:
    """Station 1 (ASSUME): versioned causal DAG + declared absent edges.

    The single source of truth (plan P2). Everything downstream is compiled
    from this object.
    """

    edges: list[tuple[str, str]]
    observed: list[str]
    latent: list[str]
    absent_edges: list[tuple[str, str]] = field(default_factory=list)
    node_roles: dict[str, str] = field(default_factory=dict)  # e.g. treatment/outcome/mediator/collider/instrument
    rationale: dict[str, str] = field(default_factory=dict)   # edge -> evidence/rationale
    version: str = ""

    def __post_init__(self) -> None:
        if not self.version:
            self.version = graph_hash(self.edges)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IdentificationProof:
    """Station 2 (IDENTIFY): criterion, adjustment set, or non-identifiability."""

    estimand_name: str
    criterion: str  # "back-door" | "front-door" | "iv" | "not-identified"
    identified: bool
    adjustment_set: list[str] = field(default_factory=list)
    instrument: str | None = None
    estimand_formula: str = ""
    graph_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataContract:
    """Station 3 (DATA): schema, provenance, and the overlap report."""

    source: str
    n_rows: int
    columns: list[str]
    regime: str
    overlap: dict[str, float] = field(default_factory=dict)  # propensity range per arm
    positivity_ok: bool = True
    graph_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureSpec:
    """Station 4 (FEATURE): what enters the model — and what must not."""

    adjustment_set: list[str]
    excluded: dict[str, str] = field(default_factory=dict)  # variable -> reason (collider/mediator/...)
    instruments: list[str] = field(default_factory=list)
    negative_controls: list[str] = field(default_factory=list)
    graph_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EstimateBundle:
    """Station 5 (MODEL): estimate + uncertainty + provenance."""

    estimand_name: str
    estimator: str
    estimate: float
    ci_low: float
    ci_high: float
    se: float
    nuisance_models: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)
    graph_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Station 6 (EVALUATE): sensitivity & robustness."""

    estimand_name: str
    e_value: float | None = None
    risk_ratio: float | None = None
    balance: dict[str, float] = field(default_factory=dict)  # max SMD etc.
    notes: list[str] = field(default_factory=list)
    graph_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RefuterResult:
    name: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalTestSuite:
    """Station 7 (TEST): refutation results + loop-invariant checks."""

    refuters: list[RefuterResult] = field(default_factory=list)
    invariant_checks: list[RefuterResult] = field(default_factory=list)
    all_green: bool = False
    graph_version: str = ""

    def finalize(self) -> None:
        self.all_green = all(r.passed for r in self.refuters + self.invariant_checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "refuters": [asdict(r) for r in self.refuters],
            "invariant_checks": [asdict(r) for r in self.invariant_checks],
            "all_green": self.all_green,
            "graph_version": self.graph_version,
        }


@dataclass
class EvolutionLogEntry:
    """Station 8 (EVOLVE): one observation of the world vs. the model."""

    check: str
    status: str  # "ok" | "alarm"
    metric: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    graph_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UCLRunReport:
    """The complete artifact chain of one UCL pass (stations 0–7)."""

    estimand: EstimandSpec
    graph: AssumptionGraph
    identification: IdentificationProof
    data: DataContract
    features: FeatureSpec
    estimate: EstimateBundle
    evaluation: EvaluationReport
    tests: CausalTestSuite

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimand": self.estimand.to_dict(),
            "graph": self.graph.to_dict(),
            "identification": self.identification.to_dict(),
            "data": self.data.to_dict(),
            "features": self.features.to_dict(),
            "estimate": self.estimate.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "tests": self.tests.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
