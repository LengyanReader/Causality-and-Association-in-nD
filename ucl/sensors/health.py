"""UCL per-station sensors extracted as standalone, graph-compiled checks
(plan section 2; section 12 task: extract ucl/sensors/).

Sensors are compilable from the AssumptionGraph — no hand-wiring.
They emit a dict with `ok: bool` and `detail` for every stage.
"""

from __future__ import annotations

from ucl.contracts.artifacts import AssumptionGraph, EstimandSpec
from ucl.graph_utils import descendants, on_causal_paths, to_nx


def check_collider_exclusion(feature_spec, graph: AssumptionGraph) -> dict:
    colliders = {c for c in feature_spec.excluded if "collider" in feature_spec.excluded[c].lower()}
    true_colliders = {c for c in feature_spec.adjustment_set
                      if c in descendants(to_nx(graph), "T") & descendants(to_nx(graph), "Y")}
    return {"ok": not true_colliders,
            "detail": {"declared_colliders": sorted(colliders),
                       "missed_colliders_in_adjustment": sorted(true_colliders)}}


def check_positivity(data_contract) -> dict:
    return {"ok": data_contract.positivity_ok, "detail": data_contract.overlap}


def check_balance(evaluation_report, threshold: float = 0.1) -> dict:
    ok = evaluation_report.balance.get("max_abs_smd", float("inf")) < threshold
    return {"ok": ok, "detail": evaluation_report.balance}


def check_sensitivity(evaluation_report, e_value_min: float = 1.5) -> dict:
    ev = evaluation_report.e_value or 1.0
    return {"ok": ev >= e_value_min, "detail": {"e_value": ev, "threshold": e_value_min}}


def check_testable_implications(graph: AssumptionGraph, findings: list[dict]) -> dict:
    violated = [f for f in findings if f.get("violated")]
    return {"ok": not violated, "detail": {"n_violated": len(violated), "pairs": [f["pair"] for f in violated]}}


def all_sensors(report, graph: AssumptionGraph,
                evolution_findings: list[dict] | None = None) -> dict[str, dict]:
    return {
        "positivity": check_positivity(report.data),
        "balance": check_balance(report.evaluation),
        "sensitivity": check_sensitivity(report.evaluation),
        "testable_implications": check_testable_implications(
            graph, evolution_findings or []),
    }
