"""Math Bridge — Level 0: Foundations (plan section 6, Level 0).

Topics: probability & Bayes' rule by simulation; Simpson's paradox (association
reverses, causation explains why); Frisch-Waugh-Lovell partialling-out (the
linear-algebra engine later used by DML orthogonality).

Rung check: everything here is rung-1 (association). Nothing in this file can
answer a do() question — that is the point.

Run:  python notebooks/math_bridge/level0_foundations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rng = np.random.default_rng(0)


def part1_bayes_rule_by_simulation() -> None:
    print("=" * 64)
    print("PART 1 — Bayes' rule by simulation")
    print("=" * 64)
    # Classic: prevalence 1%, sensitivity 99%, specificity 95%
    n = 1_000_000
    sick = rng.binomial(1, 0.01, n)
    positive = np.where(sick == 1,
                        rng.binomial(1, 0.99, n),
                        rng.binomial(1, 0.05, n))
    ppv_sim = sick[positive == 1].mean()
    # analytic: P(S|+) = .99*.01 / (.99*.01 + .05*.99)
    ppv_exact = 0.99 * 0.01 / (0.99 * 0.01 + 0.05 * 0.99)
    print(f"P(sick|positive) simulated : {ppv_sim:.4f}")
    print(f"P(sick|positive) analytic  : {ppv_exact:.4f}")
    assert abs(ppv_sim - ppv_exact) < 0.005
    print("takeaway: conditioning is just counting on a filtered population.\n")


def part2_simpsons_paradox() -> None:
    print("=" * 64)
    print("PART 2 — Simpson's paradox on NomNom-flavored data")
    print("=" * 64)
    # Within EACH city, notifications help. Aggregated, they look harmful —
    # because the platform sends most notifications in the low-order city.
    # city A (high baseline): P(Y)=0.60 control, 0.70 treated, 20% treated
    # city B (low baseline) : P(Y)=0.20 control, 0.30 treated, 80% treated
    n = 200_000
    city_b = rng.binomial(1, 0.5, n)
    treated = np.where(city_b == 1, rng.binomial(1, 0.8, n), rng.binomial(1, 0.2, n))
    base = np.where(city_b == 1, 0.2, 0.6)
    y = rng.binomial(1, base + 0.1 * treated)

    agg = y[treated == 1].mean() - y[treated == 0].mean()
    by_city = {
        c: y[(treated == 1) & (city_b == c)].mean() - y[(treated == 0) & (city_b == c)].mean()
        for c in (0, 1)
    }
    print(f"naive aggregated difference     : {agg:+.3f}  (looks harmful or null!)")
    print(f"within city A (high baseline)   : {by_city[0]:+.3f}")
    print(f"within city B (low baseline)    : {by_city[1]:+.3f}  (both +0.10)")
    assert agg < 0.05 and by_city[0] > 0.07 and by_city[1] > 0.07
    print("takeaway: the data alone cannot tell you whether to condition on")
    print("'city'. That decision is CAUSAL (is city a confounder or a mediator?),")
    print("not statistical (Pearl 2009, ch. 6; LR section 3).\n")


def part3_fwl_partialling_out() -> None:
    print("=" * 64)
    print("PART 3 — Frisch-Waugh-Lovell: regression is projection")
    print("=" * 64)
    # y = 2*t + 1.5*x + noise; the OLS coefficient on t equals the coefficient
    # from regressing residualized y on residualized t (residuals on x).
    n = 50_000
    x = rng.normal(size=(n, 2))
    t = 0.7 * x @ [1.0, -0.5] + rng.normal(size=n)
    y = 2.0 * t + 1.5 * x @ [1.0, 1.0] + rng.normal(size=n)

    # full regression
    X = np.column_stack([t, x, np.ones(n)])
    beta_full = np.linalg.lstsq(X, y, rcond=None)[0][0]

    # FWL: residualize both y and t on [x, 1]
    Xc = np.column_stack([x, np.ones(n)])
    res_t = t - Xc @ np.linalg.lstsq(Xc, t, rcond=None)[0]
    res_y = y - Xc @ np.linalg.lstsq(Xc, y, rcond=None)[0]
    beta_fwl = res_t @ res_y / (res_t @ res_t)

    print(f"coefficient on t, full OLS      : {beta_full:.4f}")
    print(f"coefficient on t, FWL           : {beta_fwl:.4f}  (truth: 2.0)")
    assert abs(beta_full - beta_fwl) < 1e-8 and abs(beta_fwl - 2.0) < 0.02
    print("takeaway: 'controlling for x' = projecting x out. Replace the linear")
    print("projection with any ML learner + cross-fitting and you have the engine")
    print("of double/debiased ML (Chernozhukov et al. 2018; LR section 5.3).")
    print("BUT: partialling out only removes LINEAR association with observed x —")
    print("it cannot create exogeneity. Orthogonality != causation.\n")


if __name__ == "__main__":
    part1_bayes_rule_by_simulation()
    part2_simpsons_paradox()
    part3_fwl_partialling_out()
    print("LEVEL 0 COMPLETE — all checks passed.")
    print("Rung reached: 1 (association). The do-operator does not exist here yet.")
