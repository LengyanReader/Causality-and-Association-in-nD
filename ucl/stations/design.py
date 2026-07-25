"""UCL stations 0–4: FRAME, ASSUME, IDENTIFY, DATA, FEATURE."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nomnom import dgp as nomnom_dgp
from nomnom.graph import nomnom_graph
from nomnom.regimes import get as get_regime
from ucl import graph_utils
from ucl.contracts.artifacts import (
    AssumptionGraph,
    DataContract,
    EstimandSpec,
    FeatureSpec,
    IdentificationProof,
)


def frame() -> EstimandSpec:
    """Station 0: the causal question, target-trial style."""
    return EstimandSpec(
        name="nomnom_notify_ate",
        question="Do push notifications cause orders on NomNom Eats?",
        treatment="T",
        outcome="Y",
        estimand="ATE",
        rung=2,
        population="all active user-days",
        decision_context="whether to keep, expand, or throttle the notification policy",
    )


def assume() -> AssumptionGraph:
    """Station 1: the versioned causal DAG (single source of truth)."""
    return nomnom_graph()


def identify(graph: AssumptionGraph, spec: EstimandSpec) -> IdentificationProof:
    """Station 2: back-door identification compiled from the graph."""
    adj = graph_utils.find_adjustment_set(graph, spec.treatment, spec.outcome)
    if adj is None:
        return IdentificationProof(
            estimand_name=spec.name,
            criterion="not-identified",
            identified=False,
            graph_version=graph.version,
        )
    adj_str = ", ".join(sorted(adj))
    return IdentificationProof(
        estimand_name=spec.name,
        criterion="back-door",
        identified=True,
        adjustment_set=sorted(adj),
        estimand_formula=f"E[Y|do(T)] = Σ_z E[Y|T, z]·P(z), z = {{{adj_str}}}",
        graph_version=graph.version,
    )


def load_data(
    proof: IdentificationProof,
    regime_name: str = "static",
    n: int = 20_000,
    seed: int = 0,
    positivity_clip: tuple[float, float] = (0.01, 0.99),
) -> tuple[pd.DataFrame, DataContract]:
    """Station 3: sample data and run the positivity/overlap check."""
    regime = get_regime(regime_name)
    df = nomnom_dgp.sample(n, regime=regime, seed=seed).drop(columns=["U"])  # U stays latent
    if proof.identified:
        X = df[proof.adjustment_set].to_numpy()
        ps = LogisticRegression(max_iter=1000).fit(X, df["T"]).predict_proba(X)[:, 1]
        t = df["T"].to_numpy()
        overlap = {
            "ps_min_treated": float(ps[t == 1].min()),
            "ps_max_treated": float(ps[t == 1].max()),
            "ps_min_control": float(ps[t == 0].min()),
            "ps_max_control": float(ps[t == 0].max()),
        }
        lo, hi = positivity_clip
        ok = bool((ps.min() >= lo) and (ps.max() <= hi))
    else:
        overlap, ok = {}, False
    contract = DataContract(
        source=f"nomnom.dgp.sample(regime={regime.name}, n={n}, seed={seed})",
        n_rows=len(df),
        columns=list(df.columns),
        regime=regime.name,
        overlap=overlap,
        positivity_ok=ok,
        graph_version=proof.graph_version,
    )
    return df, contract


def compile_features(graph: AssumptionGraph, proof: IdentificationProof) -> FeatureSpec:
    """Station 4: adjustment set in; colliders and mediators out — compiled, not hand-picked."""
    excluded: dict[str, str] = {}
    for c in sorted(graph_utils.colliders_of(graph, "T", "Y")):
        excluded[c] = "collider of T and Y - conditioning opens a spurious path"
    for m in sorted(graph_utils.on_causal_paths(graph, "T", "Y")):
        if m not in proof.adjustment_set:
            excluded[m] = "mediator on the T->Y causal path - adjusting biases the total effect"
    return FeatureSpec(
        adjustment_set=list(proof.adjustment_set),
        excluded=excluded,
        instruments=[v for v, r in graph.node_roles.items() if r == "instrument"],
        negative_controls=[v for v, r in graph.node_roles.items() if "negative_control" in r],
        graph_version=graph.version,
    )
