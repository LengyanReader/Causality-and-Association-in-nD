"""Unit tests for the NomNom DGP: ground truth sanity (causal_ci/unit)."""

import numpy as np

from nomnom.dgp import DEFAULT_PARAMS, STATIC, ground_truth, sample


def test_observational_matches_dgp_shape():
    df = sample(5_000, seed=1)
    assert len(df) == 5_000
    assert set(df["T"].unique()) <= {0, 1}
    assert set(df["Y"].unique()) <= {0, 1}
    # U must be present for testing but is excluded at the DATA station
    assert "U" in df.columns


def test_confounding_is_present_and_positive():
    """Naive (unadjusted) association must overstate the true effect."""
    df = sample(100_000, seed=2)
    naive = df.loc[df["T"] == 1, "Y"].mean() - df.loc[df["T"] == 0, "Y"].mean()
    truth = ground_truth(n_mc=200_000, seed=999)
    assert naive > truth["ate"] + 0.01  # upward confounding bias


def test_ground_truth_heterogeneity_ordering():
    """Loyal users respond more strongly than new users (plan §5.3)."""
    truth = ground_truth(n_mc=200_000, seed=999)
    assert truth["cate_loyal"] > truth["cate_new"] > 0
    # ATE lies between the two segment CATEs (convex combination)
    assert truth["cate_new"] < truth["ate"] < truth["cate_loyal"]


def test_instrument_is_randomized():
    """Z must be independent of everything it did not cause (exclusion)."""
    df = sample(100_000, seed=3)
    # Z ⊥ W (send-time jitter is randomized, W is user history)
    assert abs(np.corrcoef(df["Z"], df["W"])[0, 1]) < 0.02
    # Z actually shifts T (relevance)
    assert df.loc[df.Z == 1, "T"].mean() - df.loc[df.Z == 0, "T"].mean() > 0.2


def test_negative_control_shares_confounder_but_not_treatment():
    """NC correlates with U and with T *only through* confounding."""
    df = sample(100_000, seed=4)
    assert abs(np.corrcoef(df["NC"], df["U"])[0, 1]) > 0.25
    # within levels of U's proxy W, raw T–NC association is strongly attenuated
    df = df.copy()
    df["W_bin"] = (df["W"] > df["W"].median()).astype(int)
    raw = abs(df.groupby("W_bin").apply(
        lambda g: g.loc[g["T"] == 1, "NC"].mean() - g.loc[g["T"] == 0, "NC"].mean(),
        include_groups=False,
    )).max()
    total = abs(df.loc[df["T"] == 1, "NC"].mean() - df.loc[df["T"] == 0, "NC"].mean())
    assert raw < total


def test_holiday_regime_changes_mechanism_not_truth_structure():
    from nomnom.dgp import HOLIDAY

    truth_static = ground_truth(regime=STATIC, n_mc=200_000, seed=999)
    truth_holiday = ground_truth(regime=HOLIDAY, n_mc=200_000, seed=999)
    # both regimes have positive effects, but they differ (mechanism changed)
    assert truth_static["ate"] > 0 and truth_holiday["ate"] > 0
    assert abs(truth_static["ate"] - truth_holiday["ate"]) > 0.005
