"""Math Bridge — Level 4: estimation theory (plan section 6, Level 4).

From identified functional to estimator. Level 3 gave us the *target*
(E[Y|do(T)] via back-door). This level is about the *error budget of hitting
it with finite data* — and why naive machine learning fails at it in nD.

  Part 1: three estimators of the same functional — plug-in (g-computation),
          IPW, AIPW — all consistent, different bias/variance profile.
  Part 2: double robustness demonstrated: AIPW survives ONE wrong nuisance
          model, plug-in and IPW do not.
  Part 3: the nD trap. In high dimensions, a regularized ML plug-in is biased
          *because* it regularizes; Neyman orthogonality + cross-fitting (DML)
          removes that bias. This is the mathematical core of the project's
          "nD" theme (Chernozhukov et al. 2018; LR section 5.3).

Run:  python notebooks/math_bridge/level4_estimation_theory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nomnom.dgp import ground_truth, sample  # noqa: E402

ADJ = ["weekend", "rain", "payday", "W"]


def part1_three_estimators(df: pd.DataFrame, truth: dict) -> None:
    print("=" * 64)
    print("PART 1 — one functional, three estimators")
    print("=" * 64)
    X, T, Y = df[ADJ].to_numpy(float), df["T"].to_numpy(float), df["Y"].to_numpy(float)

    # (a) plug-in / g-computation
    out = GradientBoostingClassifier().fit(np.column_stack([X, T]), Y)
    mu1 = out.predict_proba(np.column_stack([X, np.ones(len(X))]))[:, 1].mean()
    mu0 = out.predict_proba(np.column_stack([X, np.zeros(len(X))]))[:, 1].mean()
    plugin = mu1 - mu0

    # (b) IPW / Horvitz-Thompson
    ps = GradientBoostingClassifier().fit(X, T).predict_proba(X)[:, 1].clip(0.01, 0.99)
    ipw = ((T * Y / ps).mean() - ((1 - T) * Y / (1 - ps)).mean())

    # (c) AIPW (no cross-fitting here — contrast with Part 3)
    m1 = out.predict_proba(np.column_stack([X, np.ones(len(X))]))[:, 1]
    m0 = out.predict_proba(np.column_stack([X, np.zeros(len(X))]))[:, 1]
    aipw = (m1 - m0 + T * (Y - m1) / ps - (1 - T) * (Y - m0) / (1 - ps)).mean()

    print(f"plug-in (g-computation) : {plugin:+.4f}")
    print(f"IPW (Horvitz-Thompson)  : {ipw:+.4f}")
    print(f"AIPW                    : {aipw:+.4f}")
    print(f"ground truth            : {truth['ate']:+.4f}")
    for name, est in [("plugin", plugin), ("ipw", ipw), ("aipw", aipw)]:
        assert abs(est - truth["ate"]) < 0.03, name
    print("Same estimand, three routes — all consistent HERE, where every model")
    print("is (approximately) right. The differences show up when models err.\n")


def _aipw_from(ps: np.ndarray, m1: np.ndarray, m0: np.ndarray,
               T: np.ndarray, Y: np.ndarray) -> float:
    ps = ps.clip(0.01, 0.99)
    return float((m1 - m0 + T * (Y - m1) / ps - (1 - T) * (Y - m0) / (1 - ps)).mean())


def part2_double_robustness(df: pd.DataFrame, truth: dict) -> None:
    print("=" * 64)
    print("PART 2 — double robustness: surviving one wrong model")
    print("=" * 64)
    X, T, Y = df[ADJ].to_numpy(float), df["T"].to_numpy(float), df["Y"].to_numpy(float)
    n = len(X)
    good_ps = GradientBoostingClassifier().fit(X, T)
    good_out = GradientBoostingRegressor().fit(np.column_stack([X, T]), Y)
    # "wrong" models: fit on a single covariate only (misspecified)
    bad_ps = GradientBoostingClassifier().fit(X[:, :1], T)
    bad_out = GradientBoostingRegressor().fit(np.column_stack([X[:, :1], T]), Y)

    def ps_of(model, bad: bool) -> np.ndarray:
        return model.predict_proba(X[:, :1] if bad else X)[:, 1]

    def m_of(model, bad: bool, t: float) -> np.ndarray:
        base = X[:, :1] if bad else X
        return model.predict(np.column_stack([base, np.full(n, t)]))

    rows = {}
    for ps_bad in (False, True):
        for out_bad in (False, True):
            key = f"AIPW  (ps {'BAD ' if ps_bad else 'good'}, out {'BAD' if out_bad else 'good'})"
            rows[key] = _aipw_from(
                ps_of(bad_ps if ps_bad else good_ps, ps_bad),
                m_of(bad_out if out_bad else good_out, out_bad, 1.0),
                m_of(bad_out if out_bad else good_out, out_bad, 0.0),
                T, Y,
            )
    for k, v in rows.items():
        print(f"  {k} : {v:+.4f}   (truth {truth['ate']:+.4f})")
    assert abs(rows["AIPW  (ps BAD , out good)"] - truth["ate"]) < 0.03
    assert abs(rows["AIPW  (ps good, out BAD)"] - truth["ate"]) < 0.03
    print("One nuisance model can be completely wrong and AIPW still lands on")
    print("target — the influence-function correction term absorbs the error.")
    print("(Both wrong => biased, as the fourth row shows. DR buys you one")
    print("model's worth of insurance, not two.)\n")


def part3_the_nD_trap() -> None:
    print("=" * 64)
    print("PART 3 — the nD trap: regularization bias vs. Neyman orthogonality")
    print("=" * 64)
    from sklearn.model_selection import KFold

    def one_rep(seed: int, n: int = 4_000, p: int = 200, k: int = 5, tau: float = 0.5):
        rng = np.random.default_rng(seed)
        gamma_x = np.zeros(p); gamma_x[:k] = rng.uniform(0.8, 1.2, k)
        X = rng.normal(size=(n, p))
        T = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ gamma_x - 1)))).astype(float)
        # nonlinear outcome surface the Lasso cannot represent
        m = (X[:, :k] + X[:, :k] ** 2 + np.sin(2 * X[:, :k])).sum(axis=1)
        Y = m + tau * T + 0.5 * rng.normal(size=n)

        # (a) naive plug-in: regularized LINEAR model — shrinkage + misspecification
        lasso = LassoCV(cv=3, max_iter=3000).fit(np.column_stack([X, T]), Y)
        plugin = (lasso.predict(np.column_stack([X, np.ones(n)]))
                  - lasso.predict(np.column_stack([X, np.zeros(n)]))).mean()

        # (b) DML: orthogonal score, cross-fitted flexible nuisances
        res_y, res_t = np.zeros(n), np.zeros(n)
        for tr, te in KFold(2, shuffle=True, random_state=seed).split(X):
            ly = GradientBoostingRegressor().fit(X[tr], Y[tr])
            lt = GradientBoostingClassifier().fit(X[tr], T[tr])
            res_y[te] = Y[te] - ly.predict(X[te])
            res_t[te] = T[te] - lt.predict_proba(X[te])[:, 1]
        dml = (res_t @ res_y) / (res_t @ res_t)
        se = np.sqrt(np.mean((res_y - dml * res_t) ** 2) / np.mean(res_t**2) ** 2 / n)
        return plugin - tau, dml - tau, se

    print(f"p = 200 covariates (k = 5 real, nonlinear outcome), n = 4000, "
          f"true tau = 0.5 — 3 replications")
    print(f"  {'rep':>3}  {'plug-in bias':>13}  {'DML bias':>9}  {'DML se':>7}")
    p_err, d_err = [], []
    for seed in range(3):
        pe, de, se = one_rep(seed)
        p_err.append(abs(pe)); d_err.append(abs(de))
        print(f"  {seed:>3}  {pe:>+13.4f}  {de:>+9.4f}  {se:>7.4f}")
    mp, md = np.mean(p_err), np.mean(d_err)
    print(f"  mean |bias|  plug-in {mp:.4f}  vs  DML {md:.4f}")
    assert mp > 2 * md, "plug-in should be systematically worse across reps"
    assert md < 0.06, "DML should be nearly unbiased"
    print("The Lasso MUST shrink coefficients to work in p >> n — and that")
    print("shrinkage leaks straight into the causal plug-in. The orthogonal")
    print("score (residual-on-residual, FWL from Level 0 generalized to ML)")
    print("makes the estimand insensitive to nuisance regularization: first-")
    print("order errors cancel, only second-order products remain. THAT is the")
    print("theorem that lets modern ML inside causal inference in nD.")


if __name__ == "__main__":
    df = sample(30_000, seed=42).drop(columns=["U"])
    truth = ground_truth(n_mc=200_000, seed=999)
    part1_three_estimators(df, truth)
    part2_double_robustness(df, truth)
    part3_the_nD_trap()
    print("\nLEVEL 4 COMPLETE — all checks passed.")
    print("Identification says WHAT is computable; estimation theory says HOW")
    print("WELL finite data can compute it — and in nD, only orthogonal scores")
    print("keep the two error budgets separate.")
