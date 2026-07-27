"""UCL per-station actuators extracted as standalone revision actions
(plan section 2; section 12 task: extract ucl/actuators/).

Each actuator is a pure function: (failed sensor, current state) -> next action.
They encode the controller policy priority queue from the plan:

  (i) fix failed invariants,
  (ii) fix failed refuters,
  (iii) fix positivity,
  (iv) reduce sensitivity (E-value),
  (v) reduce variance.
"""

from __future__ import annotations

from ucl.contracts.artifacts import UCLRunReport


def escalate_identification(report: UCLRunReport) -> str:
    return (
        "ACTION: estimand not identified under the current graph. "
        "Options: (a) add an instrument edge, (b) accept a back-door set "
        "via stronger assumptions, (c) route to causal discovery (station 8)."
    )


def trim_to_common_support(overlap: dict, threshold: float = 0.01) -> str:
    lo = min(overlap.get("ps_min_treated", 0), overlap.get("ps_min_control", 0))
    return (
        f"ACTION: trim observations with propensity < {threshold}. "
        f"Current min PS = {lo:.3f}. This changes the estimand to ATT on the "
        f"overlapping population — document that narrowing."
    )


def strengthen_sensitivity(bundle, e_value: float, target: float = 2.0) -> str:
    return (
        f"ACTION: E-value = {e_value:.1f} < target. "
        f"Options: (a) collect a negative-control outcome, "
        f"(b) find an instrument for a sensitivity-free estimate, "
        f"(c) report the E-value as-is with a fragility caveat."
    )


def revise_assumption_graph(
    refuter_name: str, detail: dict, graph_version: str
) -> str:
    return (
        f"ACTION: refuter '{refuter_name}' fired on graph v{graph_version}. "
        f"Detail: {detail}. Open an assumption revision (station 1) — "
        f"the failing refuter localizes which edge or mechanism is suspect."
    )


def re_run_on_new_regime(drift_localized: str) -> str:
    return (
        f"ACTION: mechanism drift localized to '{drift_localized}'. "
        f"Re-run the full UCL pass (station 5 → 6 → 7) on the new data batch "
        f"and record the new graph-grounding evidence."
    )
