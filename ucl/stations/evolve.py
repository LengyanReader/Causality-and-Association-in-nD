"""UCL station 8: EVOLVE (plan section 3 — the meta-loop).

Three monitors, all compiled from the AssumptionGraph:

1. Mechanism-stability monitor (invariance principle; Peters et al. 2016, LR
   section 6): fit P(node | parents) on the reference regime; a correctly
   specified causal mechanism must travel to new data. The node whose
   conditional degrades most localizes the drift.
2. Testable-implication monitor: absent edges imply (conditional)
   independencies; marginal ones are tested on the new batch.
3. Negative-control alarm: adjusted T -> NC effect must stay null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from ucl.contracts.artifacts import EvolutionLogEntry, EstimandSpec, FeatureSpec
from ucl.graph_utils import _d_separated, to_nx
from ucl.stations.analysis import aipw_crossfit

# latent parents are replaced by their observed proxy (U -> W in NomNom)
PROXY = {"U": "W"}

# nodes with a purely deterministic assignment — no stochastic mechanism to monitor
DETERMINISTIC = {"coupon"}

# alarm thresholds
MECHANISM_ALARM_NATS = 0.02   # log-loss degradation beyond this = mechanism changed
MEAN_SHIFT_ALARM_Z = 4.0      # |z| for exogenous marginal shifts


def observed_parents(graph, node: str) -> list[str]:
    parents = [a for a, b in graph.edges if b == node]
    out = []
    for p in parents:
        if p in graph.observed:
            out.append(p)
        elif p in PROXY and PROXY[p] != node:
            out.append(PROXY[p])
    return sorted(set(out) - {node})


def mechanism_stability(
    graph,
    df_ref: pd.DataFrame,
    df_new: pd.DataFrame,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Log-loss degradation per mechanism: ref holdout vs. new regime."""
    results: dict[str, dict[str, float]] = {}
    for node in sorted(set(graph.observed) - DETERMINISTIC):
        parents = observed_parents(graph, node)
        is_binary = (
            df_ref[node].nunique() <= 2 and df_new[node].nunique() <= 2
            and set(df_ref[node].unique()) <= {0, 1}
        )
        if not parents or not is_binary:
            # exogenous node: compare marginal means (binary or continuous)
            x0, x1 = df_ref[node].to_numpy(float), df_new[node].to_numpy(float)
            pooled = np.sqrt(x0.var(ddof=1) / len(x0) + x1.var(ddof=1) / len(x1))
            z = 0.0 if pooled == 0 else abs(x0.mean() - x1.mean()) / pooled
            results[node] = {"kind": "marginal", "z": float(z)}
            continue
        if df_ref[node].nunique() < 2 or df_new[node].nunique() < 2:
            continue
        X0 = df_ref[parents].to_numpy(float)
        y0 = df_ref[node].to_numpy()
        X_tr, X_ho, y_tr, y_ho = train_test_split(
            X0, y0, test_size=0.25, random_state=seed, stratify=y0
        )
        clf = GradientBoostingClassifier().fit(X_tr, y_tr)
        loss_ho = log_loss(y_ho, clf.predict_proba(X_ho), labels=[0, 1])
        loss_new = log_loss(df_new[node].to_numpy(),
                            clf.predict_proba(df_new[parents].to_numpy(float)),
                            labels=[0, 1])
        results[node] = {
            "kind": "mechanism",
            "loss_ref_holdout": float(loss_ho),
            "loss_new": float(loss_new),
            "degradation": float(loss_new - loss_ho),
        }
    return results


def testable_implications(graph, df: pd.DataFrame, alpha: float = 0.001) -> list[dict]:
    """Test marginally-implied independencies of declared absent edges."""
    g = to_nx(graph)
    findings = []
    for a, b in graph.absent_edges:
        if a not in graph.observed or b not in graph.observed:
            continue
        if not _d_separated(g, {a}, {b}, set()):
            continue  # only marginal implications are tested in v1
        x, y = df[a].to_numpy(float), df[b].to_numpy(float)
        r, p = stats.pearsonr(x, y)
        findings.append({
            "pair": f"{a} _|_ {b}",
            "implied_by": f"absent edge {a}-/->{b}",
            "r": float(r),
            "p_value": float(p),
            "violated": bool(p < alpha),
        })
    return findings


def negative_control_alarm(
    df: pd.DataFrame,
    spec: EstimandSpec,
    features: FeatureSpec,
    seed: int = 0,
) -> list[EvolutionLogEntry]:
    entries = []
    for nc in features.negative_controls:
        if nc not in df.columns:
            continue
        res = aipw_crossfit(df, spec.treatment, nc, features.adjustment_set, seed=seed)
        tol = max(0.01, 1.96 * res["se"])
        entries.append(EvolutionLogEntry(
            check=f"negative_control:{nc}",
            status="ok" if abs(res["ate"]) < tol else "alarm",
            metric=res["ate"],
            detail={"tolerance": tol},
            graph_version=features.graph_version,
        ))
    return entries


def evolve(
    graph,
    spec: EstimandSpec,
    features: FeatureSpec,
    df_ref: pd.DataFrame,
    df_new: pd.DataFrame,
    seed: int = 0,
) -> tuple[list[EvolutionLogEntry], dict]:
    """Full EVOLVE pass: monitors + localization. Returns (log entries, report)."""
    entries: list[EvolutionLogEntry] = []

    stability = mechanism_stability(graph, df_ref, df_new, seed=seed)
    mech = {n: r for n, r in stability.items() if r["kind"] == "mechanism"}
    marg = {n: r for n, r in stability.items() if r["kind"] == "marginal"}

    worst = max(mech, key=lambda n: mech[n]["degradation"]) if mech else None
    drift_detected = bool(worst and mech[worst]["degradation"] > MECHANISM_ALARM_NATS)
    for n, r in sorted(mech.items()):
        alarm = r["degradation"] > MECHANISM_ALARM_NATS
        entries.append(EvolutionLogEntry(
            check=f"mechanism_stability:{n}",
            status="alarm" if alarm else "ok",
            metric=r["degradation"],
            detail=r,
            graph_version=graph.version,
        ))
    for n, r in sorted(marg.items()):
        entries.append(EvolutionLogEntry(
            check=f"marginal_shift:{n}",
            status="alarm" if r["z"] > MEAN_SHIFT_ALARM_Z else "ok",
            metric=r["z"],
            detail=r,
            graph_version=graph.version,
        ))

    implications = testable_implications(graph, df_new)
    for f in implications:
        entries.append(EvolutionLogEntry(
            check=f"testable_implication:{f['pair']}",
            status="alarm" if f["violated"] else "ok",
            metric=f["p_value"],
            detail=f,
            graph_version=graph.version,
        ))

    entries.extend(negative_control_alarm(df_new, spec, features, seed=seed))

    report = {
        "drift_detected": drift_detected,
        "localized_mechanism": worst if drift_detected else None,
        "stability": stability,
        "implications": implications,
        "n_alarms": sum(1 for e in entries if e.status == "alarm"),
    }
    return entries, report
