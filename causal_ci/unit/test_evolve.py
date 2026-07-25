"""Unit tests for the EVOLVE station (plan section 3, P3 acceptance):

- control batch (same regime) must NOT alarm;
- holiday batch must alarm AND localize to the M mechanism;
- testable implications hold on static data;
- the actuator (re-run on holiday) recovers the holiday truth.
"""

import pytest

from nomnom.dgp import HOLIDAY, STATIC, ground_truth, sample
from nomnom.graph import nomnom_graph
from ucl.engine import run_pass
from ucl.stations import compile_features, frame, identify
from ucl.stations.evolve import (
    MECHANISM_ALARM_NATS,
    evolve,
    mechanism_stability,
)
from ucl.stations.evolve import testable_implications as check_implications


@pytest.fixture(scope="module")
def setup():
    graph = nomnom_graph()
    spec = frame()
    proof = identify(graph, spec)
    features = compile_features(graph, proof)
    ref = sample(10_000, regime=STATIC, seed=100).drop(columns=["U"])
    ctrl = sample(10_000, regime=STATIC, seed=200).drop(columns=["U"])
    drift = sample(10_000, regime=HOLIDAY, seed=300).drop(columns=["U"])
    return graph, spec, features, ref, ctrl, drift


def test_control_batch_no_mechanism_alarm(setup):
    graph, spec, features, ref, ctrl, _ = setup
    _, report = evolve(graph, spec, features, ref, ctrl, seed=0)
    assert not report["drift_detected"]
    assert report["localized_mechanism"] is None


def test_holiday_batch_alarms_and_localizes_to_M(setup):
    graph, spec, features, ref, _, drift = setup
    _, report = evolve(graph, spec, features, ref, drift, seed=0)
    assert report["drift_detected"]
    assert report["localized_mechanism"] == "M"
    # the changed mechanism dominates all others by a wide margin
    mech = {n: r for n, r in report["stability"].items() if r["kind"] == "mechanism"}
    others = [r["degradation"] for n, r in mech.items() if n != "M"]
    assert mech["M"]["degradation"] > 3 * max(others)
    # and the causal mechanism of interest stayed invariant
    assert mech["Y"]["degradation"] < MECHANISM_ALARM_NATS


def test_seasonal_shift_flagged_as_marginal_not_mechanism(setup):
    """Rain probability changes in holiday — a marginal (parent) shift, not a
    mechanism change. The monitor must report it as a marginal_shift alarm."""
    graph, spec, features, ref, _, drift = setup
    stab = mechanism_stability(graph, ref, drift, seed=0)
    assert stab["rain"]["kind"] == "marginal" and stab["rain"]["z"] > 4.0
    assert stab["T"]["kind"] == "mechanism"
    assert stab["T"]["degradation"] < MECHANISM_ALARM_NATS


def test_testable_implications_hold_on_static_data(setup):
    graph, *_ , ref, ctrl, drift = setup
    findings = check_implications(graph, ref)
    assert findings  # at least some marginal implications were testable
    assert not any(f["violated"] for f in findings)


def test_actuator_recover_holiday_truth():
    """Full meta-loop: after detection, re-run pass on holiday data."""
    report, _ = run_pass(regime="holiday", n=20_000, seed=23)
    truth_h = ground_truth(regime=HOLIDAY, n_mc=200_000, seed=999)
    assert report.tests.all_green
    assert report.estimate.ci_low <= truth_h["ate"] <= report.estimate.ci_high
