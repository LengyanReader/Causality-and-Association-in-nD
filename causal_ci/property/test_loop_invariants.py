"""Property tests: loop invariants that must hold for EVERY run (plan section 2).

I1: every artifact carries the current graph version.
I2: no adjustment variable is a descendant of treatment.
I3: evaluation report exists with recorded sensitivity parameters.
I4: positivity coverage and post-weighting balance within thresholds.
"""

import pytest

from nomnom.dgp import HOLIDAY, ground_truth, sample
from ucl.engine import run_pass
from ucl.stations import frame, identify
from ucl.stations.design import assume

TRUTH = ground_truth(n_mc=200_000, seed=999)
TRUTH_HOLIDAY = ground_truth(regime=HOLIDAY, n_mc=200_000, seed=999)


@pytest.mark.parametrize(
    "seed,n,smd_tol",
    [
        (21, 20_000, 0.10),  # strict: full-size sample meets the classic 0.1 rule
        (22, 8_000, 0.15),   # small-n: finite-sample noise inflates weighted SMD
    ],
)
def test_loop_invariants_hold_across_seeds(seed, n, smd_tol):
    report, _ = run_pass(regime="static", n=n, seed=seed)
    # I1 — engine asserts internally; verify externally too
    assert report.identification.graph_version == report.graph.version
    assert report.features.graph_version == report.graph.version
    assert report.estimate.graph_version == report.graph.version
    assert report.tests.graph_version == report.graph.version
    # I3
    assert report.evaluation.e_value is not None
    # I4
    assert report.data.positivity_ok
    assert report.evaluation.balance["max_abs_smd"] < smd_tol


def test_holiday_regime_pass_satisfies_invariants_and_recovers_truth():
    """The same graph identifies in the holiday regime (structure unchanged);
    the pipeline must recover the HOLIDAY truth — mechanisms may shift,
    structure does not (plan section 5.2)."""
    report, _ = run_pass(regime="holiday", n=20_000, seed=23)
    assert report.tests.all_green
    assert report.estimate.ci_low <= TRUTH_HOLIDAY["ate"] <= report.estimate.ci_high


def test_adjustment_set_never_contains_treatment_descendants():
    """I2 as a graph property, independent of any dataset."""
    graph = assume()
    proof = identify(graph, frame())
    from ucl.graph_utils import descendants, to_nx

    desc = descendants(to_nx(graph), "T")
    assert not (set(proof.adjustment_set) & desc)
