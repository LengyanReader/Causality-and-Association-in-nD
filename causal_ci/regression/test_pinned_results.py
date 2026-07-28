"""Regression tests: pinned numerical results (plan section 4.3, regression layer).

Values pinned 2026-07-25 from the deterministic DGP (fixed n_mc and seed).
If any of these move, either the DGP or an estimator changed -- the diff must
be reviewed and the pins re-based deliberately.

Tolerances are set for cross-platform reproducibility: NumPy Generator RNG is
bit-identical across platforms with the same seed, but GradientBoosting (and
other sklearn ensemble methods) may have minor cross-platform variance in
tree-building due to BLAS/OpenMP differences.
"""

import pytest

from nomnom.dgp import HOLIDAY, STATIC, ground_truth, sample
from ucl.engine import run_pass
from ucl.stations.analysis import aipw_crossfit

# --- pinned DGP ground truths (n_mc=200_000, seed=999) ---
PIN_STATIC = {"ate": 0.2413, "mu1": 0.61324, "mu0": 0.37194,
              "cate_loyal": 0.29912, "cate_new": 0.15501}
PIN_HOLIDAY = {"ate": 0.19098, "mu1": 0.57548, "mu0": 0.3845}

# --- pinned pipeline result (run_pass static, n=20_000, seed=0) ---
PIN_STATIC_PASS_ATE = 0.2437

# Cross-platform tolerances: ground-truth is deterministic (NumPy RNG),
# ensemble methods may have minor platform variance in tree building.
TRUTH_TOL = 0.01       # ground truth pins (RNG is bit-identical)
PASS_TOL = 0.015       # full UCL pass (includes GradientBoosting)
AIPW_TOL = 0.03        # raw AIPW estimator


@pytest.mark.parametrize(
    "regime,pin",
    [(STATIC, PIN_STATIC), (HOLIDAY, PIN_HOLIDAY)],
)
def test_ground_truth_pins(regime, pin):
    truth = ground_truth(regime=regime, n_mc=200_000, seed=999)
    for key, val in pin.items():
        assert abs(truth[key] - val) < TRUTH_TOL, (
            f"{regime.name}.{key}: {truth[key]} vs pinned {val}")


def test_static_pass_estimate_pin():
    report, _ = run_pass(regime="static", n=20_000, seed=0)
    assert abs(report.estimate.estimate - PIN_STATIC_PASS_ATE) < PASS_TOL
    assert report.tests.all_green


def test_aipw_estimator_pin():
    """Deterministic estimator path: fixed data seed and model seed."""
    df = sample(20_000, seed=42).drop(columns=["U"])
    res = aipw_crossfit(df, "T", "Y", ["weekend", "rain", "payday", "W"], seed=42)
    assert abs(res["ate"] - PIN_STATIC["ate"]) < AIPW_TOL
    assert 0 < res["se"] < 0.02
