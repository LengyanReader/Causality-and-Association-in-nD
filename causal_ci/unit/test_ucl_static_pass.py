"""Unit tests for the UCL static pass: ground-truth recovery + refuters.

These are the causal-CI acceptance tests (plan §8, P1):
- the identified, cross-fit AIPW pipeline recovers the true ATE;
- refuters pass on a valid analysis and FAIL on planted violations.
"""

import numpy as np
import pandas as pd
import pytest

from nomnom.dgp import ground_truth, sample
from nomnom.graph import nomnom_graph
from ucl import graph_utils
from ucl.contracts.artifacts import EstimandSpec
from ucl.engine import run_pass
from ucl.stations import compile_features, frame, identify
from ucl.stations.analysis import aipw_crossfit
from ucl.stations.analysis import test_suite as run_test_suite


@pytest.fixture(scope="module")
def truth():
    return ground_truth(n_mc=200_000, seed=999)


@pytest.fixture(scope="module")
def spec():
    return frame()


def test_identification_finds_valid_backdoor_set():
    graph = nomnom_graph()
    proof = identify(graph, frame())
    assert proof.identified
    assert proof.criterion == "back-door"
    # W must be in the adjustment set — without it the U-path is open
    assert "W" in proof.adjustment_set
    # collider S, mediator M, instrument Z must never appear
    for bad in ("S", "M", "Z", "NC"):
        assert bad not in proof.adjustment_set


def test_features_exclude_collider_and_mediator():
    graph = nomnom_graph()
    proof = identify(graph, frame())
    features = compile_features(graph, proof)
    assert "S" in features.excluded
    assert "M" in features.excluded
    assert features.instruments == ["Z"]
    assert features.negative_controls == ["NC"]


def test_aipw_recovers_ground_truth(truth, spec):
    df = sample(30_000, seed=7).drop(columns=["U"])
    adj = ["weekend", "rain", "payday", "W"]
    res = aipw_crossfit(df, spec.treatment, spec.outcome, adj, seed=7)
    assert abs(res["ate"] - truth["ate"]) < 3 * res["se"]


def test_omitting_proxy_confounder_biases_estimate(truth, spec):
    """Without W the U-confounding path is open: estimate must be off."""
    df = sample(30_000, seed=8).drop(columns=["U"])
    res = aipw_crossfit(df, spec.treatment, spec.outcome, ["weekend", "rain", "payday"], seed=8)
    assert abs(res["ate"] - truth["ate"]) > 3 * res["se"]


def test_collider_adjustment_corrupts_estimate(truth, spec):
    """Adjusting for the collider S must change/bias the estimate (LR §3)."""
    df = sample(30_000, seed=9).drop(columns=["U"])
    good = aipw_crossfit(df, spec.treatment, spec.outcome, ["weekend", "rain", "payday", "W"], seed=9)
    bad = aipw_crossfit(df, spec.treatment, spec.outcome, ["weekend", "rain", "payday", "W", "S"], seed=9)
    assert abs(bad["ate"] - truth["ate"]) > abs(good["ate"] - truth["ate"])


def test_full_pass_all_green_and_covers_truth(truth):
    report, evolution = run_pass(regime="static", n=20_000, seed=0)
    assert report.tests.all_green
    assert report.estimate.ci_low <= truth["ate"] <= report.estimate.ci_high
    assert report.evaluation.balance["max_abs_smd"] < 0.1


def test_refuters_detect_planted_bias(truth, spec):
    """Run the refutation battery on a *broken* feature spec (no W): the
    negative-control refuter must fire — residual confounding shows up on NC."""
    df = sample(20_000, seed=10).drop(columns=["U"])
    graph = nomnom_graph()
    broken = compile_features(graph, identify(graph, spec))
    broken.adjustment_set = [v for v in broken.adjustment_set if v != "W"]
    suite = run_test_suite(df, spec, broken, evaluation=None, graph=graph, seed=10)
    nc_results = [r for r in suite.refuters if r.name.startswith("negative_control")]
    assert nc_results and not all(r.passed for r in nc_results)


def test_coverage_across_seeds(truth):
    """Acceptance criterion (plan §8 P1): CI covers truth in ~95% of runs."""
    covers = 0
    n_runs = 25  # smaller than 100 for CI speed; acceptance run uses 100
    for seed in range(n_runs):
        df = sample(20_000, seed=100 + seed).drop(columns=["U"])
        res = aipw_crossfit(df, "T", "Y", ["weekend", "rain", "payday", "W"], seed=seed)
        covers += res["ci"][0] <= truth["ate"] <= res["ci"][1]
    assert covers / n_runs >= 0.84  # binomial slack around 0.95
