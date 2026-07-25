"""UCL stations 5–7: MODEL (cross-fit AIPW/DML), EVALUATE, TEST.

The estimator is AIPW with 2-fold cross-fitting (Chernozhukov et al. 2018,
LR §5.3): Neyman-orthogonal score, so nuisance-model errors enter only at
second order. E-value follows VanderWeele & Ding (2017, LR §8.6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

from ucl.contracts.artifacts import (
    CausalTestSuite,
    EstimateBundle,
    EstimandSpec,
    EvaluationReport,
    FeatureSpec,
    RefuterResult,
)

_EPS = 0.01


def _clip(ps: np.ndarray) -> np.ndarray:
    return np.clip(ps, _EPS, 1 - _EPS)


def aipw_crossfit(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjust: list[str],
    n_folds: int = 2,
    seed: int = 0,
) -> dict:
    """Cross-fitted AIPW (doubly robust) estimate of the ATE for binary T, Y."""
    X = df[adjust].to_numpy(dtype=float)
    T = df[treatment].to_numpy(dtype=float)
    Y = df[outcome].to_numpy(dtype=float)
    n = len(df)
    e = np.zeros(n)
    m0 = np.zeros(n)
    m1 = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train, test in kf.split(X):
        ps_model = GradientBoostingClassifier().fit(X[train], T[train])
        e[test] = ps_model.predict_proba(X[test])[:, 1]
        out_model = GradientBoostingClassifier().fit(
            np.column_stack([X[train], T[train]]), Y[train]
        )
        m1[test] = out_model.predict_proba(np.column_stack([X[test], np.ones(test.size)]))[:, 1]
        m0[test] = out_model.predict_proba(np.column_stack([X[test], np.zeros(test.size)]))[:, 1]
    e = _clip(e)
    mu1 = m1 + T * (Y - m1) / e
    mu0 = m0 + (1 - T) * (Y - m0) / (1 - e)
    psi = mu1 - mu0
    est = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(n))
    return {
        "ate": est,
        "se": se,
        "ci": (est - 1.96 * se, est + 1.96 * se),
        "mu1": float(mu1.mean()),
        "mu0": float(mu0.mean()),
        "e": e,
        "diagnostics": {
            "e_mean_treated": float(e[T == 1].mean()),
            "e_mean_control": float(e[T == 0].mean()),
        },
    }


def model(
    df: pd.DataFrame,
    spec: EstimandSpec,
    features: FeatureSpec,
    seed: int = 0,
) -> EstimateBundle:
    """Station 5: estimate the identified estimand with cross-fit AIPW."""
    res = aipw_crossfit(df, spec.treatment, spec.outcome, features.adjustment_set, seed=seed)
    return EstimateBundle(
        estimand_name=spec.name,
        estimator="AIPW cross-fit (DML, GradientBoosting nuisances)",
        estimate=res["ate"],
        se=res["se"],
        ci_low=res["ci"][0],
        ci_high=res["ci"][1],
        nuisance_models={
            "propensity": "GradientBoostingClassifier",
            "outcome": "GradientBoostingClassifier(T-augmented)",
        },
        diagnostics=res["diagnostics"],
        graph_version=features.graph_version,
    )


def _smd(x: np.ndarray, t: np.ndarray, w: np.ndarray) -> float:
    """Weighted standardized mean difference of one covariate across arms."""
    t = t.astype(bool)
    w1, w0 = w[t], w[~t]
    x1, x0 = x[t], x[~t]
    mu1 = np.average(x1, weights=w1)
    mu0 = np.average(x0, weights=w0)
    var = (np.average((x1 - mu1) ** 2, weights=w1) + np.average((x0 - mu0) ** 2, weights=w0)) / 2
    return float(0.0 if var <= 0 else (mu1 - mu0) / np.sqrt(var))


def evaluate(
    df: pd.DataFrame,
    spec: EstimandSpec,
    features: FeatureSpec,
    bundle: EstimateBundle,
) -> EvaluationReport:
    """Station 6: E-value (VanderWeele & Ding 2017) + IPW balance check."""
    res = aipw_crossfit(df, spec.treatment, spec.outcome, features.adjustment_set)
    rr = res["mu1"] / res["mu0"] if res["mu0"] > 0 else np.nan
    rr_ev = rr if rr >= 1 else 1.0 / rr
    e_value = float(rr_ev + np.sqrt(rr_ev * (rr_ev - 1))) if np.isfinite(rr_ev) and rr_ev > 1 else 1.0
    e = res["e"]
    T = df[spec.treatment].to_numpy()
    w = T / e + (1 - T) / (1 - e)
    balance = {
        f"max_abs_smd": max(
            abs(_smd(df[c].to_numpy(dtype=float), T, w)) for c in features.adjustment_set
        )
    }
    notes = []
    if balance["max_abs_smd"] >= 0.1:
        notes.append("Balance alarm: max |SMD| >= 0.1 after weighting")
    if e_value < 1.5:
        notes.append("Fragility warning: E-value < 1.5 — modest unmeasured confounding could explain the effect")
    return EvaluationReport(
        estimand_name=spec.name,
        e_value=e_value,
        risk_ratio=float(rr),
        balance=balance,
        notes=notes,
        graph_version=bundle.graph_version,
    )


def _refute_placebo_treatment(df, spec, features, seed):
    """Permute T within the data; a valid pipeline must estimate ~0."""
    rng = np.random.default_rng(seed + 1)
    df_p = df.copy()
    df_p[spec.treatment] = rng.permutation(df_p[spec.treatment].to_numpy())
    res = aipw_crossfit(df_p, spec.treatment, spec.outcome, features.adjustment_set, seed=seed)
    tol = max(0.01, 1.96 * res["se"])
    return RefuterResult(
        name="placebo_treatment",
        passed=abs(res["ate"]) < tol,
        detail={"estimate": res["ate"], "tolerance": tol},
    )


def _refute_random_common_cause(df, spec, features, seed):
    """Add a random confounder; the estimate must be stable."""
    rng = np.random.default_rng(seed + 2)
    df_r = df.copy()
    df_r["rcc"] = rng.normal(size=len(df_r))
    base = aipw_crossfit(df, spec.treatment, spec.outcome, features.adjustment_set, seed=seed)
    perturbed = aipw_crossfit(
        df_r, spec.treatment, spec.outcome, features.adjustment_set + ["rcc"], seed=seed
    )
    delta = abs(perturbed["ate"] - base["ate"])
    tol = max(0.005, 1.0 * base["se"])
    return RefuterResult(
        name="random_common_cause",
        passed=delta < tol,
        detail={"delta": delta, "tolerance": tol},
    )


def _refute_subset(df, spec, features, seed, frac=0.8):
    """Estimate on an 80% subset; must agree with the full-sample estimate."""
    rng = np.random.default_rng(seed + 3)
    sub = df.sample(frac=frac, random_state=int(rng.integers(1e6)))
    full = aipw_crossfit(df, spec.treatment, spec.outcome, features.adjustment_set, seed=seed)
    part = aipw_crossfit(sub, spec.treatment, spec.outcome, features.adjustment_set, seed=seed)
    delta = abs(part["ate"] - full["ate"])
    tol = 2.0 * part["se"]
    return RefuterResult(
        name="subset_refuter",
        passed=delta < tol,
        detail={"delta": delta, "tolerance": tol},
    )


def _refute_negative_control(df, spec, features, nc, seed):
    """Adjusted effect of T on the negative-control outcome must be ~null."""
    res = aipw_crossfit(df, spec.treatment, nc, features.adjustment_set, seed=seed)
    tol = max(0.01, 1.96 * res["se"])
    return RefuterResult(
        name=f"negative_control:{nc}",
        passed=abs(res["ate"]) < tol,
        detail={"estimate": res["ate"], "tolerance": tol},
    )


def test_suite(
    df: pd.DataFrame,
    spec: EstimandSpec,
    features: FeatureSpec,
    evaluation: EvaluationReport | None,
    graph,
    seed: int = 0,
) -> CausalTestSuite:
    """Station 7: refutation battery + loop-invariant checks (plan §2, §4.3)."""
    suite = CausalTestSuite(graph_version=features.graph_version)
    suite.refuters.append(_refute_placebo_treatment(df, spec, features, seed))
    suite.refuters.append(_refute_random_common_cause(df, spec, features, seed))
    suite.refuters.append(_refute_subset(df, spec, features, seed))
    for nc in features.negative_controls:
        if nc in df.columns:
            suite.refuters.append(_refute_negative_control(df, spec, features, nc, seed))
    # Loop invariant 2: no adjustment variable is a descendant of T
    from ucl.graph_utils import descendants, to_nx

    desc_t = descendants(to_nx(graph), spec.treatment)
    bad = [v for v in features.adjustment_set if v in desc_t]
    suite.invariant_checks.append(
        RefuterResult(
            name="invariant2:no_descendant_adjustment",
            passed=not bad,
            detail={"violators": bad},
        )
    )
    # Loop invariant 3: evaluation report exists with recorded sensitivity params
    suite.invariant_checks.append(
        RefuterResult(
            name="invariant3:evaluation_recorded",
            passed=evaluation is not None and evaluation.e_value is not None,
            detail={},
        )
    )
    suite.finalize()
    return suite
