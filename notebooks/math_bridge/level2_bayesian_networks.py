"""Math Bridge — Level 2: Bayesian networks (plan section 6, Level 2).

Topics: the Markov condition and d-separation on the NomNom graph; verifying
graph-implied independencies empirically; structure learning with a mini-PC
algorithm (constraint-based skeleton recovery).

BRIDGE INSIGHT: a Bayesian network is a *joint distribution with a graph*.
The same observational distribution factorizes over every DAG in the same
Markov equivalence class — so arrows in a BN are not (yet) causal claims.
Orientation needs assumptions beyond observational CI structure.

Run:  python notebooks/math_bridge/level2_bayesian_networks.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nomnom.dgp import sample  # noqa: E402
from nomnom.graph import nomnom_graph  # noqa: E402
from ucl.graph_utils import _d_separated, to_nx  # noqa: E402

ALPHA = 0.001


def part1_dseparation_explorer() -> None:
    print("=" * 64)
    print("PART 1 — d-separation explorer on the NomNom graph")
    print("=" * 64)
    g = to_nx(nomnom_graph())
    queries = [
        ({"Z"}, {"W"}, set(), "Z (jitter) _|_ W (app history)"),
        ({"Z"}, {"rain"}, set(), "Z _|_ rain"),
        ({"Z"}, {"Y"}, set(), "Z _|_ Y (marginally)"),
        ({"Z"}, {"Y"}, {"T"}, "Z _|_ Y | T"),
        ({"rain"}, {"Z"}, {"S"}, "rain _|_ Z | S (S is a collider descendant!)"),
        ({"coupon"}, {"T"}, set(), "coupon _|_ T"),
        ({"M"}, {"Y"}, {"T"}, "M _|_ Y | T (direct T->Y edge keeps them linked?)"),
    ]
    for x, y, z, label in queries:
        sep = _d_separated(g, x, y, z)
        cond = f" | {','.join(sorted(z))}" if z else ""
        print(f"  {label:55s} d-separated: {sep}")
    # key checks
    assert _d_separated(g, {"Z"}, {"W"}, set())
    assert not _d_separated(g, {"Z"}, {"Y"}, set())
    # conditioning on S (a descendant of the collider T->S<-Y) OPENS a path
    assert _d_separated(g, {"rain"}, {"Z"}, set())
    assert not _d_separated(g, {"rain"}, {"Z"}, {"S"})
    print("takeaway: conditioning can DESTROY independence (colliders).")
    print("Whether a variable helps or harms is a graph property, not a")
    print("statistical one.\n")


def _chi2_p(x: np.ndarray, y: np.ndarray) -> float:
    return stats.chi2_contingency(pd.crosstab(x, y))[1]


def _cmh_p(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Stratified chi-square (Cochran-Mantel-Haenszel style) for binary x,y | z."""
    stat, df = 0.0, 0
    for zv in np.unique(z):
        m = z == zv
        if m.sum() < 30 or len(np.unique(x[m])) < 2 or len(np.unique(y[m])) < 2:
            continue
        tab = pd.crosstab(x[m], y[m]).to_numpy(float)
        row, col = tab.sum(1, keepdims=True), tab.sum(0, keepdims=True)
        exp = row @ col / tab.sum()
        stat += ((tab - exp) ** 2 / np.maximum(exp, 1e-9)).sum()
        df += (tab.shape[0] - 1) * (tab.shape[1] - 1)
    return float(1 - stats.chi2.cdf(stat, df)) if df else 1.0


def part2_empirical_verification(df: pd.DataFrame) -> None:
    print("=" * 64)
    print("PART 2 — empirical verification of graph-implied (in)dependencies")
    print("=" * 64)
    checks = [
        ("Z", "W", None, True), ("Z", "Y", None, False),
        ("coupon", "T", None, True), ("rain", "payday", None, True),
        ("T", "NC", None, False),        # linked via W<-U->NC confounding
        ("weekend", "Z", "T", False),    # T is a collider: conditioning opens the path
    ]
    for a, b, cond, expect_indep in checks:
        p = _chi2_p(df[a], df[b]) if cond is None else _cmh_p(df[a], df[b], df[cond])
        indep = p > ALPHA
        label = f"{a} _|_ {b}" + (f" | {cond}" if cond else "")
        print(f"  {label:24s} p={p:.4g}  -> {'independent' if indep else 'dependent'} "
              f"(expected {'indep' if expect_indep else 'dep'})")
        assert indep == expect_indep, label
    # the same Berkson effect, continuous version: W and Z are marginally
    # independent but become NEGATIVELY correlated within T strata
    marg = np.corrcoef(df["W"], df["Z"])[0, 1]
    within = [np.corrcoef(df.loc[df["T"] == t, "W"], df.loc[df["T"] == t, "Z"])[0, 1]
              for t in (0, 1)]
    print(f"  corr(W, Z) marginal        : {marg:+.3f}")
    print(f"  corr(W, Z | T=0), (T=1)    : {within[0]:+.3f}, {within[1]:+.3f}")
    assert abs(marg) < 0.02 and all(c < -0.05 for c in within)
    print("takeaway: the Markov condition is TESTABLE — the graph makes")
    print("falsifiable predictions about the joint distribution. And conditioning")
    print("on a collider (Berkson's bias) manufactures dependence from nothing.\n")


def part3_mini_pc(df: pd.DataFrame) -> None:
    print("=" * 64)
    print("PART 3 — mini-PC: skeleton recovery from CI tests")
    print("=" * 64)
    vars_ = ["weekend", "rain", "payday", "Z", "T", "M", "Y", "S", "NC", "segment", "coupon"]
    d = df[vars_]
    adj = {frozenset(e) for e in combinations(vars_, 2)}
    # level 0: marginal tests
    for x, y in combinations(vars_, 2):
        if frozenset((x, y)) in adj and _chi2_p(d[x], d[y]) > ALPHA:
            adj.discard(frozenset((x, y)))
    # level 1: single-variable conditioning
    for x, y in combinations(vars_, 2):
        if frozenset((x, y)) not in adj:
            continue
        neighbors = [z for z in vars_
                     if z not in (x, y)
                     and (frozenset((x, z)) in adj or frozenset((y, z)) in adj)]
        for z in neighbors:
            if _cmh_p(d[x], d[y], d[z]) > ALPHA:
                adj.discard(frozenset((x, y)))
                break
    true_edges = {
        frozenset(e) for e in [
            ("weekend", "T"), ("weekend", "Y"), ("rain", "T"), ("rain", "Y"),
            ("payday", "T"), ("payday", "Y"), ("Z", "T"), ("T", "M"), ("M", "Y"),
            ("T", "Y"), ("coupon", "Y"), ("T", "S"), ("Y", "S"), ("segment", "Y"),
        ]
    }
    found_true = adj & true_edges
    spurious = adj - true_edges
    missed = true_edges - adj
    print(f"true skeleton edges found : {len(found_true)}/{len(true_edges)}")
    print(f"missed                    : {sorted(tuple(sorted(e)) for e in missed)}")
    print(f"spurious                  : {sorted(tuple(sorted(e)) for e in spurious)}")
    assert len(found_true) >= 12  # most of the skeleton recovered
    assert any("NC" in e for e in spurious), (
        "latent U should leave spurious NC edges — this is WHY FCI exists (LR 6)"
    )
    print("takeaway 1: CI tests recover much of the skeleton from data alone.")
    print("takeaway 2: the latent hunger U leaves spurious NC edges — constraint-")
    print("based discovery without latent variables (PC) hits its limit here;")
    print("FCI exists precisely for this case (Spirtes et al. 2000).\n")


def part4_markov_equivalence(df: pd.DataFrame) -> None:
    print("=" * 64)
    print("PART 4 — orientation limit: Markov equivalence")
    print("=" * 64)
    # Chain T->M->Y and chain Y->M->T imply the SAME CI structure:
    # T _|/ Y marginally, T _|_ Y | M... but NomNom has a direct T->Y edge,
    # so use the Z->T->M subchain instead.
    p_marg = _chi2_p(df["Z"], df["M"])
    p_cond = _cmh_p(df["Z"], df["M"], df["T"])
    print(f"Z _|_ M ?        p={p_marg:.4g} (dependent)")
    print(f"Z _|_ M | T ?    p={p_cond:.4g} (independent)")
    assert p_marg < ALPHA and p_cond > ALPHA
    print("These two facts are consistent with Z->T->M AND with Z<-T<-M — no")
    print("reversal of that chain changes the implied independencies. The joint")
    print("distribution factorizes over every DAG in the equivalence class.")
    print("\nA BN's arrows are factorization structure, not mechanisms. Giving them")
    print("causal meaning = promoting the DAG to a STRUCTURAL CAUSAL MODEL —")
    print("that promotion is Level 3.")


if __name__ == "__main__":
    part1_dseparation_explorer()
    df = sample(50_000, seed=42).drop(columns=["U"])
    part2_empirical_verification(df)
    part3_mini_pc(df)
    part4_markov_equivalence(df)
    print("\nLEVEL 2 COMPLETE — all checks passed.")
    print("Rung reached: 1, but with graph structure — the launchpad for rung 2.")
