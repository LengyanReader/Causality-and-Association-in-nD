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


def test_content_numbers_match_pipeline_output():
    """Regression: content strings must not contain stale hardcoded numbers.

    The demo narrative text must derive from the same pipeline output as the
    live data.  Any stale hardcoded literal in the content means a number
    drifted out of sync on regeneration — a content bug.
    """
    import json, sys
    from pathlib import Path

    data_path = Path(__file__).resolve().parents[2] / "docs" / "demo" / "data.json"
    if not data_path.exists():
        pytest.skip("data.json not found — run python docs/demo/generate.py first")

    d = json.loads(data_path.read_text(encoding="utf-8"))

    live = d["static"]
    evolve = d.get("evolve", {})

    # Map of (stale_value, live_getter, tolerance_check, label)
    checks = [
        # naive ATE: was 0.343 or 0.345; live is live["naive"]
        ("0.343", lambda: f"{live['naive']:.3f}", None, "naive ATE (0.343)"),
        ("0.345", lambda: f"{live['naive']:.3f}", None, "naive ATE (0.345)"),
        # confounding gap: was 0.102 or 0.104; live is live["gap"]
        ("0.102", lambda: f"{live['gap']:.3f}", None, "confounding gap (0.102)"),
        ("0.104", lambda: f"{live['gap']:.3f}", None, "confounding gap (0.104)"),
        # E-value: was 2.73; live is live["e_value"]
        ("2.73", lambda: f"{live['e_value']:.2f}", None, "E-value (2.73)"),
    ]

    # Search all content strings
    content_text = json.dumps(d["content"])

    failures = []
    for stale, getter, _, label in checks:
        if stale in content_text:
            failures.append(f"STALE: '{stale}' ({label}) still in content — should be '{getter()}'")

    if failures:
        pytest.fail("\n".join(failures))
