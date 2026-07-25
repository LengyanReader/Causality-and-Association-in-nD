"""Math Bridge — Level 3: the causal step (plan section 6, Level 3).

Promoting a Bayesian network to a STRUCTURAL CAUSAL MODEL: the arrows become
autonomous mechanisms, and a new operator appears — do().

  Part 1: P(Y|T) vs P(Y|do(T)) — truncated factorization by hand.
  Part 2: back-door adjustment from scratch == g-computation == truth.
  Part 3: front-door criterion CHECKED FIRST — and shown to fail here,
          because the graph has a direct T->Y edge (criteria are graph
          theorems, not recipes).
  Part 4: rung 3 — counterfactuals by abduction-action-prediction on the
          SCM itself, and how they differ from interventional quantities.

Run:  python notebooks/math_bridge/level3_causal_step.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import nomnom.dgp as dgp  # noqa: E402
from nomnom.dgp import ground_truth, sample  # noqa: E402
from nomnom.graph import nomnom_graph  # noqa: E402
from ucl.graph_utils import on_causal_paths  # noqa: E402

ADJ = ["weekend", "rain", "payday", "W"]


def part1_conditioning_vs_intervening(df: pd.DataFrame, truth: dict) -> None:
    print("=" * 64)
    print("PART 1 — P(Y|T) vs P(Y|do(T)): truncated factorization by hand")
    print("=" * 64)
    naive = df.loc[df["T"] == 1, "Y"].mean() - df.loc[df["T"] == 0, "Y"].mean()
    print(f"P(Y=1|T=1) - P(Y=1|T=0)        : {naive:+.4f}   (observational)")
    print(f"E[Y|do(T=1)] - E[Y|do(T=0)]    : {truth['ate']:+.4f}   (ground truth)")
    print("Truncated factorization: do(T=t) DELETES the equation for T and")
    print("replaces P(T|parents) with a point mass at t. Every other mechanism")
    print("is untouched — that is graph surgery, and it is all the do-operator is.")
    assert naive > truth["ate"] + 0.05
    print("The gap between the two rows IS confounding, made numerical.\n")


def part2_backdoor_from_scratch(df: pd.DataFrame, truth: dict) -> None:
    print("=" * 64)
    print("PART 2 — back-door g-computation from scratch")
    print("=" * 64)
    # E[Y|do(T=t)] = sum_z E[Y | T=t, Z=z] * P(z), z = {weekend, rain, payday, W}
    X = df[ADJ + ["T"]].to_numpy(float)
    y = df["Y"].to_numpy()
    model = GradientBoostingClassifier().fit(X, y)
    X1 = df[ADJ].assign(T=1).to_numpy(float)
    X0 = df[ADJ].assign(T=0).to_numpy(float)
    mu1 = model.predict_proba(X1)[:, 1].mean()   # average over the OBSERVED P(z)
    mu0 = model.predict_proba(X0)[:, 1].mean()
    est = mu1 - mu0
    print(f"g-computation ATE estimate     : {est:+.4f}")
    print(f"ground truth                   : {truth['ate']:+.4f}")
    print(f"E[Y|do(T=1)] = {mu1:.4f} (truth {truth['mu1']:.4f})   "
          f"E[Y|do(T=0)] = {mu0:.4f} (truth {truth['mu0']:.4f})")
    assert abs(est - truth["ate"]) < 0.02
    print("Same observational data as Part 1 — different OPERATOR. The formula")
    print("weights by P(z), not P(z|T): that is the whole difference between")
    print("seeing and doing (Pearl 2009, thm 3.2.2).\n")


def part3_frontdoor_criteria_are_theorems(df: pd.DataFrame, truth: dict) -> None:
    print("=" * 64)
    print("PART 3 — front-door: check the criterion BEFORE applying it")
    print("=" * 64)
    graph = nomnom_graph()
    mediators = on_causal_paths(graph, "T", "Y")
    direct_edge = ("T", "Y") in graph.edges
    print(f"mediators on T->Y paths: {sorted(mediators)}")
    print(f"direct T->Y edge exists: {direct_edge}")
    assert direct_edge
    # front-door formula (would be valid only if M intercepted ALL T->Y paths):
    # E[Y|do(T=t)] = sum_m P(m|T=t) * sum_t' E[Y|m, T=t'] P(T=t')
    p_m = df.groupby("T")["M"].mean()
    e_y = df.groupby(["M", "T"])["Y"].mean()
    p_t = df["T"].mean()
    fd1 = sum(p_m[1] * e_y.get((1, tp), 0.0) * (p_t if tp else 1 - p_t) for tp in (0, 1))
    fd0 = sum((1 - p_m[1]) * e_y.get((0, tp), 0.0) * (p_t if tp else 1 - p_t) for tp in (0, 1))
    est_fd = fd1 - fd0
    print(f"front-door estimate          : {est_fd:+.4f}")
    print(f"ground truth                 : {truth['ate']:+.4f}")
    print("...deceptively close! Criterion violations do not announce themselves.")
    # Push the direct effect up (stronger T->Y edge): the same formula,
    # applied outside its assumptions, fails catastrophically.
    strong = dgp.DGPParams(tau_new=1.2, tau_loyal=2.0)
    df_s = sample(50_000, params=strong, seed=42).drop(columns=["U"])
    truth_s = ground_truth(params=strong, n_mc=200_000, seed=999)
    p_ms = df_s.groupby("T")["M"].mean()
    e_ys = df_s.groupby(["M", "T"])["Y"].mean()
    p_ts = df_s["T"].mean()
    fd1_s = sum(p_ms[1] * e_ys.get((1, tp), 0.0) * (p_ts if tp else 1 - p_ts) for tp in (0, 1))
    fd0_s = sum((1 - p_ms[1]) * e_ys.get((0, tp), 0.0) * (p_ts if tp else 1 - p_ts) for tp in (0, 1))
    est_fd_s = fd1_s - fd0_s
    print(f"with a stronger direct edge  : front-door {est_fd_s:+.4f} vs truth {truth_s['ate']:+.4f}")
    assert abs(est_fd_s - truth_s["ate"]) > 0.1, "front-door must FAIL outside its assumptions"
    print("Front-door requires M to intercept EVERY directed T->Y path. NomNom's")
    print("direct edge violates that — so the formula quietly returns the wrong")
    print("number. Identification criteria are theorems ABOUT THE GRAPH, not")
    print("recipes: apply them only after the graph says you may (UCL station 2).\n")


def part4_counterfactuals(truth: dict) -> None:
    print("=" * 64)
    print("PART 4 — rung 3: counterfactuals by abduction-action-prediction")
    print("=" * 64)
    # Use the SCM itself (testing access): sample exogenous noise, observe the
    # factual world, then re-run the SAME noise under do(T=1-T_factual).
    n = 200_000
    rng = np.random.default_rng(555)
    exo = dgp._draw_exogenous(n, rng, dgp.STATIC, dgp.DEFAULT_PARAMS)
    factual = dgp._structural(exo, dgp.STATIC, dgp.DEFAULT_PARAMS, t_value=None)
    flip = 1 - factual["T"].to_numpy()
    cf = dgp._structural(exo, dgp.STATIC, dgp.DEFAULT_PARAMS, t_value=flip)
    # Among treated users who ordered: would they have ordered WITHOUT the nudge?
    mask = (factual["T"] == 1) & (factual["Y"] == 1)
    p_necessity = 1 - cf.loc[mask, "Y"].mean()
    # Interventional contrast (rung 2) for comparison — same exogenous draws:
    y1 = dgp._structural(exo, dgp.STATIC, dgp.DEFAULT_PARAMS, t_value=np.ones(1, int))["Y"]
    y0 = dgp._structural(exo, dgp.STATIC, dgp.DEFAULT_PARAMS, t_value=np.zeros(1, int))["Y"]
    ate = y1.mean() - y0.mean()
    print(f"P(Y_0=0 | T=1, Y=1)  probability the order was CAUSED by the nudge:")
    print(f"  counterfactual (rung 3)      : {p_necessity:.4f}")
    print(f"  interventional ATE (rung 2)  : {ate:.4f}")
    print(f"  (Monte-Carlo ground truth    : {truth['ate']:+.4f})")
    assert abs(ate - truth["ate"]) < 0.01
    assert p_necessity > ate  # necessity among affected units > average effect
    print("\nAbduction (infer the noise from the evidence) -> action (do(T)) ->")
    print("prediction (re-run the mechanisms). No interventional distribution")
    print("answers 'was THIS order caused by the nudge?' — that question lives")
    print("one rung higher, and only an SCM with its noise structure reaches it.")


if __name__ == "__main__":
    df = sample(50_000, seed=42).drop(columns=["U"])
    truth = ground_truth(n_mc=200_000, seed=999)
    part1_conditioning_vs_intervening(df, truth)
    part2_backdoor_from_scratch(df, truth)
    part3_frontdoor_criteria_are_theorems(df, truth)
    part4_counterfactuals(truth)
    print("\nLEVEL 3 COMPLETE — all checks passed.")
    print("Rung reached: 3. The full ladder: see -> do -> imagine.")
