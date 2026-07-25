"""Math Bridge — Level 1: Bayesian inference (plan section 6, Level 1).

Topics: conjugate updating (Beta-Binomial) on NomNom order rates; a
Metropolis-Hastings sampler from scratch recovering the same posterior;
posterior predictive checking; and the crucial BRIDGE INSIGHT:

    Bayesian inference updates beliefs about parameters of a FIXED
    observational model. It never leaves rung 1 by itself.

We prove the last point numerically: a perfectly-updated Bayesian posterior
over P(Y=1 | T=1) and P(Y=1 | T=0) on NomNom observational data is sharply
concentrated — on the WRONG causal effect (confounded). The posterior is not
wrong about the association; the association is not the effect.

Run:  python notebooks/math_bridge/level1_bayesian_inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nomnom.dgp import ground_truth, sample  # noqa: E402

rng = np.random.default_rng(0)


def part1_conjugate_update(df) -> None:
    print("=" * 64)
    print("PART 1 — Beta-Binomial conjugate update (order rate among treated)")
    print("=" * 64)
    a0, b0 = 2.0, 8.0  # prior: mean 0.2, weak
    treated = df[df["T"] == 1]
    k, n = int(treated["Y"].sum()), len(treated)
    a1, b1 = a0 + k, b0 + n - k
    print(f"prior           : Beta({a0}, {b0}), mean {a0/(a0+b0):.3f}")
    print(f"data            : {k} orders / {n} treated user-days")
    print(f"posterior       : Beta({a1:.0f}, {b1:.0f}), mean {a1/(a1+b1):.4f}")
    print(f"posterior 95% CI: [{stats.beta.ppf(0.025, a1, b1):.4f}, "
          f"{stats.beta.ppf(0.975, a1, b1):.4f}]")
    assert abs(a1 / (a1 + b1) - k / n) < 0.01
    return a1, b1


def part2_metropolis_hastings(a1: float, b1: float) -> None:
    print("\n" + "=" * 64)
    print("PART 2 — Metropolis-Hastings from scratch vs. the analytic posterior")
    print("=" * 64)
    log_post = lambda p: stats.beta.logpdf(p, a1, b1)
    samples, current, accepted = [], 0.3, 0
    for i in range(120_000):
        proposal = current + rng.normal(0, 0.02)
        if 0 < proposal < 1 and np.log(rng.uniform()) < log_post(proposal) - log_post(current):
            current = proposal
            accepted += 1
        if i >= 20_000 and i % 10 == 0:  # burn-in + thinning
            samples.append(current)
    samples = np.array(samples)
    ks = stats.kstest(samples, stats.beta(a1, b1).cdf).pvalue
    print(f"acceptance rate : {accepted/120_000:.2f}")
    print(f"MCMC mean       : {samples.mean():.4f}   analytic mean: {a1/(a1+b1):.4f}")
    print(f"MCMC sd         : {samples.std():.4f}   analytic sd  : {stats.beta.std(a1, b1):.4f}")
    print(f"KS test vs Beta : p = {ks:.3f} (large p = sampler matches posterior)")
    assert abs(samples.mean() - a1 / (a1 + b1)) < 0.005 and ks > 0.01


def part3_posterior_predictive_check(a1: float, b1: float, df) -> None:
    print("\n" + "=" * 64)
    print("PART 3 — Posterior predictive check")
    print("=" * 64)
    treated = df[df["T"] == 1]
    p_draws = stats.beta.rvs(a1, b1, size=2_000, random_state=1)
    rep_means = (rng.uniform(size=(2_000, 500)) < p_draws[:, None]).mean(axis=1)
    obs = treated["Y"].iloc[:500].mean()
    ok = np.quantile(rep_means, 0.025) <= obs <= np.quantile(rep_means, 0.975)
    print(f"replicated means 95% band: [{np.quantile(rep_means, 0.025):.4f}, "
          f"{np.quantile(rep_means, 0.975):.4f}]")
    print(f"observed subsample mean  : {obs:.4f}  -> {'CONSISTENT' if ok else 'MISMATCH'}")
    assert ok


def part4_the_bridge_insight(df) -> None:
    print("\n" + "=" * 64)
    print("PART 4 — THE BRIDGE INSIGHT: a perfect posterior on the wrong quantity")
    print("=" * 64)
    t1 = df[df["T"] == 1]["Y"]
    t0 = df[df["T"] == 0]["Y"]
    # razor-sharp posteriors over the two CONDITIONAL rates
    post1 = (2 + t1.sum(), 8 + len(t1) - t1.sum())
    post0 = (2 + t0.sum(), 8 + len(t0) - t0.sum())
    diff_draws = (stats.beta.rvs(*post1, size=50_000, random_state=2)
                  - stats.beta.rvs(*post0, size=50_000, random_state=3))
    naive = diff_draws.mean()
    lo, hi = np.quantile(diff_draws, [0.025, 0.975])
    truth = ground_truth(n_mc=200_000, seed=999)
    print(f"posterior over P(Y=1|T=1) - P(Y=1|T=0): {naive:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"ground-truth ATE  E[Y|do(T=1)] - E[Y|do(T=0)]: {truth['ate']:+.4f}")
    print(f"posterior sd    : {diff_draws.std():.5f}  (vanishingly confident...)")
    covers = lo <= truth["ate"] <= hi
    print(f"does the 95% posterior band contain the causal truth? {covers}")
    assert not covers, "confounding should put the truth outside the band"
    print("\nThe Bayesian machinery worked flawlessly — and answered a rung-1")
    print("question. P(Y|T) != P(Y|do(T)): no amount of posterior refinement")
    print("closes that gap. Closing it requires ASSUMPTIONS (the causal graph),")
    print("which is Level 3 of the bridge. The likelihood never sees do().")


if __name__ == "__main__":
    df = sample(30_000, seed=42).drop(columns=["U"])
    a1, b1 = part1_conjugate_update(df)
    part2_metropolis_hastings(a1, b1)
    part3_posterior_predictive_check(a1, b1, df)
    part4_the_bridge_insight(df)
    print("\nLEVEL 1 COMPLETE — all checks passed.")
    print("Rung reached: still 1. Bayes conditions; it does not intervene.")
