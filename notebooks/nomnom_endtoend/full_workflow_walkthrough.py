"""Causal Science — the full 9-station UCL walkthrough on NomNom Eats.

Styled as a narrative walkthrough script (convertible to .ipynb via
scripts/py_to_ipynb.py). Run end-to-end for the complete workflow,
every station explained.

Run:  python notebooks/nomnom_endtoend/full_workflow_walkthrough.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============================================================================
# PART 0 — SETUP
# ============================================================================
def part0_setup():
    """Import the workflow components and the ground-truth world.
    NomNom Eats: a food-delivery platform. Business question:
    do push notifications cause orders, and for whom?

    Key features of this world for our walkthrough:
      - Confounding (U, a latent hunger variable, drives both notifications
        and orders — the platform has a proxy W = measured app use)
      - A mediator (M = app open)
      - A collider (S = engagement score — never adjust for this!)
      - An instrument (Z = randomized send-time jitter)
      - A negative-control outcome (NC = battery drain)
      - Heterogeneous effects (loyal > new users)
      - Two regimes (static vs holiday, where the T→M mechanism drifts)

    Everything we estimate can be checked against ground truth.
    """
    from nomnom.dgp import ground_truth, sample
    from ucl.contracts.artifacts import AssumptionGraph, EstimandSpec, UCLRunReport

    truth = ground_truth(n_mc=200_000, seed=999)
    print(f"Ground truth ATE   : {truth['ate']:+.4f}")
    print(f"  loyal users      : {truth['cate_loyal']:+.4f}")
    print(f"  new users        : {truth['cate_new']:+.4f}")
    print(f"  (computed by Monte Carlo under do(T=1) vs do(T=0) with")
    print(f"   common random numbers)")
    assert truth["ate"] > 0 and truth["cate_loyal"] > truth["cate_new"]
    return truth


# ============================================================================
# STATION 0 — FRAME: What decision does this inform?
# ============================================================================
def station0_frame():
    """Define the causal question as an estimand, target-trial style.
    (plan section 2, station 0; LR section 8.1)
    """
    from nomnom.graph import nomnom_graph
    from ucl.stations import frame

    spec = frame()
    print(f"Question          : {spec.question}")
    print(f"Estimand          : {spec.estimand}")
    print(f"Treatment (T)     : {spec.treatment}")
    print(f"Outcome (Y)       : {spec.outcome}")
    print(f"Population        : {spec.population}")
    print(f"Rung              : {spec.rung} (intervention — this is a do() question)")
    assert spec.rung == 2 and spec.estimand == "ATE"
    return spec


# ============================================================================
# STATION 1 — ASSUME: What causal structure do we believe?
# ============================================================================
def station1_assume():
    """Declare the causal graph — our assumptions made VISIBLE.
    The graph is the single source of truth (plan P2).
    Everything downstream is compiled from it.
    """
    from nomnom.graph import nomnom_graph

    graph = nomnom_graph()
    print(f"Graph version     : {graph.version}")
    print(f"Observed nodes    : {sorted(graph.observed)}")
    print(f"Latent nodes      : {sorted(graph.latent)}")
    print(f"Edges             : {len(graph.edges)}")
    print(f"Declared absent edges : {len(graph.absent_edges)}")
    print(f"Node roles:")
    for v, r in sorted(graph.node_roles.items()):
        print(f"  {v:>8s} : {r}")
    print(f"\nKey absent-edge assumptions (the falsifiable part):")
    for a, b in graph.absent_edges:
        print(f"  {a} -/-> {b}")
    assert "T" in graph.node_roles and graph.node_roles["T"] == "treatment"
    return graph


# ============================================================================
# STATION 2 — IDENTIFY: Is the estimand computable from observables?
# ============================================================================
def station2_identify(graph, spec):
    """The back-door criterion: compiled from the graph by graph surgery.
    We DELETE all outgoing edges from T, then search for an observed set
    Z that d-separates T from Y.
    """
    from ucl.stations import identify

    proof = identify(graph, spec)
    print(f"Criterion         : {proof.criterion}")
    print(f"Identified        : {proof.identified}")
    print(f"Adjustment set    : {sorted(proof.adjustment_set)}")
    print(f"Formula           : {proof.estimand_formula}")
    assert proof.identified and "W" in proof.adjustment_set
    return proof


# ============================================================================
# STATION 3 — DATA: Do the data support the identification?
# ============================================================================
def station3_data(proof, truth):
    """Sample from the observational world and run the positivity/overlap check.
    The data station gatekeeps: if overlap fails, refuse to estimate untrimmed.
    """
    from nomnom.dgp import sample
    from ucl.stations import load_data

    df, contract = load_data(proof, regime_name="static", n=20_000, seed=0)
    print(f"Rows              : {contract.n_rows}")
    print(f"Source            : {contract.source}")
    print(f"Positivity OK     : {contract.positivity_ok}")
    for k, v in sorted(contract.overlap.items()):
        print(f"  {k:>20s}: {v:.4f}")
    assert contract.positivity_ok and len(df) == 20_000
    print("\nObservational vs. ground-truth:")
    naive = df.loc[df["T"] == 1, "Y"].mean() - df.loc[df["T"] == 0, "Y"].mean()
    print(f"  P(Y|T=1)-P(Y|T=0)   = {naive:+.4f}   (confounded)")
    print(f"  E[Y|do(T=1)]-E[Y|do(T=0)] = {truth['ate']:+.4f}   (truth)")
    print(f"  confounding gap = {naive-truth['ate']:+.4f}")
    assert naive > truth["ate"]
    return df, contract


# ============================================================================
# STATION 4 — FEATURE: What enters the model — and what must not?
# ============================================================================
def station4_features(graph, proof):
    """The feature spec is compiled — not hand-picked. Every excluded variable
    is either a collider (conditioning opens a spurious path) or a mediator
    (conditioning blocks the effect we're trying to measure).
    """
    from ucl.stations import compile_features

    features = compile_features(graph, proof)
    print(f"Adjustment set  : {sorted(features.adjustment_set)}")
    print(f"Instruments     : {features.instruments}")
    print(f"Neg. controls   : {features.negative_controls}")
    print(f"Excluded (never adjust):")
    for v, reason in sorted(features.excluded.items()):
        print(f"  {v} : {reason}")
    assert "S" in features.excluded and "M" in features.excluded
    return features


# ============================================================================
# STATION 5 — MODEL: How do we estimate?
# ============================================================================
def station5_model(df, spec, features, truth):
    """Cross-fit AIPW / Double Machine Learning.
    Neyman-orthogonal score: nuisance-model errors enter at second order,
    so we can use flexible ML without contaminating the causal estimand.
    """
    from ucl.stations import model
    from ucl.stations.analysis import aipw_crossfit

    bundle = model(df, spec, features, seed=0)
    print(f"Estimator        : {bundle.estimator}")
    print(f"ATE estimate     : {bundle.estimate:+.4f}")
    print(f"95% CI           : [{bundle.ci_low:+.4f}, {bundle.ci_high:+.4f}]")
    print(f"Ground truth     : {truth['ate']:+.4f}")
    covers = bundle.ci_low <= truth["ate"] <= bundle.ci_high
    print(f"CI covers truth  : {covers}")
    assert covers
    return bundle


# ============================================================================
# STATION 6 — EVALUATE: How wrong could we be?
# ============================================================================
def station6_evaluate(df, spec, features, bundle):
    """Sensitivity analysis: E-value + post-weighting balance.
    The E-value (VanderWeele & Ding 2017) quantifies how strong unmeasured
    confounding would need to be to explain away the result.
    """
    from ucl.stations import evaluate

    evaluation = evaluate(df, spec, features, bundle)
    print(f"E-value          : {evaluation.e_value:.2f}")
    print(f"Risk ratio       : {evaluation.risk_ratio:.2f}")
    print(f"Max |SMD|        : {evaluation.balance['max_abs_smd']:.4f}")
    for n in evaluation.notes:
        print(f"  [{n}]" if n else "")
    assert evaluation.e_value > 1.5
    assert evaluation.balance["max_abs_smd"] < 0.1
    return evaluation


# ============================================================================
# STATION 7 — TEST: Refutation battery + loop invariants
# ============================================================================
def station7_test(df, spec, features, evaluation, graph):
    """Refutation: placebo treatment, random common cause, subset refuter,
    negative-control outcome. Plus loop-invariant checks that are the
    causal equivalent of type-checking.
    """
    from ucl.stations import test_suite as run_test_suite

    suite = run_test_suite(df, spec, features, evaluation, graph, seed=0)
    print("Refuters:")
    for r in suite.refuters:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}")
    print("Loop invariants:")
    for r in suite.invariant_checks:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}")
    print(f"ALL GREEN        : {suite.all_green}")
    assert suite.all_green
    return suite


# ============================================================================
# STATION 8 — EVOLVE: Is the world still the one we modeled?
# ============================================================================
def station8_evolve(graph, spec, features, df_ref, truth):
    """The mechanism-stability monitor:
    1. fit P(node | parents) per node on the reference batch;
    2. evaluate on the new (holiday) batch;
    3. flag the mechanism whose conditional degrades most as the drift locus.

    Then the actuator: re-estimate on the new regime and check against
    the holiday ground truth.
    """
    from nomnom.dgp import HOLIDAY, ground_truth, sample
    from ucl.stations.evolve import evolve, mechanism_stability

    df_new = sample(10_000, regime=HOLIDAY, seed=300).drop(columns=["U"])
    stab = mechanism_stability(graph, df_ref, df_new, seed=0)
    mech = {n: r for n, r in stab.items() if r["kind"] == "mechanism"}
    worst = max(mech, key=lambda n: mech[n]["degradation"])

    print(f"Mechanism-stability monitor (static -> holiday):")
    for n, r in sorted(mech.items()):
        flag = " <-- DRIFT" if r["degradation"] > 0.02 else ""
        print(f"  {n:8s} degradation={r['degradation']:+.4f}{flag}")
    print(f"\nDrift detected and localized to: {worst}")
    assert worst == "M"

    # Actuator: re-run the full pass on the holiday regime
    from ucl.engine import run_pass

    holiday_report, _ = run_pass(regime="holiday", n=20_000, seed=23)
    truth_h = ground_truth(regime=HOLIDAY, n_mc=200_000, seed=999)
    covers_h = (holiday_report.estimate.ci_low <= truth_h["ate"]
                <= holiday_report.estimate.ci_high)
    print(f"Holiday ATE      : {holiday_report.estimate.estimate:+.4f} "
          f"(truth {truth_h['ate']:+.4f}, covers={covers_h})")
    print(f"Holiday refuters : {'ALL GREEN' if holiday_report.tests.all_green else 'SOME FAILED'}")
    assert covers_h and holiday_report.tests.all_green
    return worst, holiday_report


# ============================================================================
# MAIN — the complete walkthrough
# ============================================================================
def main():
    print("=" * 72)
    print("CAUSAL SCIENCE — FULL WORKFLOW WALKTHROUGH")
    print("The Universal Causal Loop on NomNom Eats")
    print("=" * 72)

    truth = part0_setup()

    print("\n" + "=" * 72)
    print("STATION 0 — FRAME")
    spec = station0_frame()

    print("\n" + "=" * 72)
    print("STATION 1 — ASSUME")
    graph = station1_assume()

    print("\n" + "=" * 72)
    print("STATION 2 — IDENTIFY")
    proof = station2_identify(graph, spec)

    print("\n" + "=" * 72)
    print("STATION 3 — DATA")
    df, contract = station3_data(proof, truth)

    print("\n" + "=" * 72)
    print("STATION 4 — FEATURE")
    features = station4_features(graph, proof)

    print("\n" + "=" * 72)
    print("STATION 5 — MODEL")
    bundle = station5_model(df, spec, features, truth)

    print("\n" + "=" * 72)
    print("STATION 6 — EVALUATE")
    evaluation = station6_evaluate(df, spec, features, bundle)

    print("\n" + "=" * 72)
    print("STATION 7 — TEST")
    suite = station7_test(df, spec, features, evaluation, graph)

    print("\n" + "=" * 72)
    print("STATION 8 — EVOLVE")
    # df already has U dropped by the DATA station; EVOLVE gets observables only
    worst, holiday_report = station8_evolve(graph, spec, features, df, truth)

    # — compile the full artifact chain —
    from ucl.contracts.artifacts import UCLRunReport

    report = UCLRunReport(
        estimand=spec, graph=graph, identification=proof,
        data=contract, features=features, estimate=bundle,
        evaluation=evaluation, tests=suite,
    )
    out_path = REPO_ROOT / "runs" / "full_walkthrough_report.json"
    out_path.write_text(report.to_json())

    print("\n" + "=" * 72)
    print("WORKFLOW COMPLETE — all 9 stations passed.")
    print(f"Static ATE      : {bundle.estimate:+.4f}  (truth {truth['ate']:+.4f})")
    print(f"Holiday drift   : localized to {worst}")
    from nomnom.dgp import HOLIDAY as _HOL, ground_truth as _gt
    _th = _gt(regime=_HOL, n_mc=200_000, seed=999)
    print(f"Holiday ATE     : {holiday_report.estimate.estimate:+.4f}  "
          f"(truth {truth['ate']:+.4f} static, {_th['ate']:+.4f} holiday)")
    print(f"E-value         : {evaluation.e_value:.1f}")
    print(f"CI tests        : {suite.all_green}")
    print(f"Report          : {out_path}")
    print("\nThe loop closed: frame → assume → identify → data → feature →")
    print("model → evaluate → test → evolve — with every artifact carrying")
    print("its graph version, every assumption visible and debatable, every")
    print("estimate unit-tested against known ground truth.")
    print("=" * 72)


if __name__ == "__main__":
    main()
