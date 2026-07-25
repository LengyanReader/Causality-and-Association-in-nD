"""Unit tests for the event-driven UCL runner (P4)."""

import pytest

from nomnom.dgp import HOLIDAY, STATIC, ground_truth, sample
from loops.event_runner import UCLEventLoop


@pytest.fixture(scope="module")
def flow():
    loop = UCLEventLoop()
    r1 = loop.emit("batch_arrived",
                   df=sample(10_000, regime=STATIC, seed=100).drop(columns=["U"]),
                   tag="static#1", seed=0)
    r2 = loop.emit("batch_arrived",
                   df=sample(10_000, regime=STATIC, seed=200).drop(columns=["U"]),
                   tag="static#2", seed=1)
    r3 = loop.emit("batch_arrived",
                   df=sample(10_000, regime=HOLIDAY, seed=300).drop(columns=["U"]),
                   tag="holiday#1", seed=2)
    truth_h = ground_truth(regime=HOLIDAY, n_mc=200_000, seed=999)
    return loop, r1, r2, r3, truth_h


def test_first_batch_establishes_baseline(flow):
    _, r1, *_ = flow
    assert r1["action"] == "baseline"
    assert r1["report"].tests.all_green


def test_same_regime_batch_no_alarm(flow):
    _, _, r2, _, _ = flow
    assert r2["action"] == "monitor_ok"
    assert not r2["evolution"]["drift_detected"]


def test_holiday_batch_triggers_rerun_and_recovers_truth(flow):
    _, _, _, r3, truth_h = flow
    assert r3["action"] == "rerun"
    assert r3["evolution"]["localized_mechanism"] == "M"
    rep = r3["report"]
    assert rep.tests.all_green
    assert rep.estimate.ci_low <= truth_h["ate"] <= rep.estimate.ci_high


def test_event_log_is_complete(flow):
    loop, *_ = flow
    kinds = [e["event"] for e in loop.log]
    assert kinds == ["baseline_established", "batch_arrived", "batch_arrived", "actuator_rerun"]
    # every report carries the same graph version (loop invariant 1)
    versions = {r.graph.version for r in loop.reports.values()}
    assert len(versions) == 1


def test_unknown_event_rejected(flow):
    loop, *_ = flow
    with pytest.raises(ValueError):
        loop.emit("definitely_not_an_event")
