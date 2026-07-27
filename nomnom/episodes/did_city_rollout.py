"""NomNom Episode #2 — staggered DiD: city-by-city notification rollout
(plan section 5.2 coverage matrix: DiD row).

The platform rolls out a notification algorithm change city by city over
several weeks; treated cities get the new algorithm; control cities stay on
the old one. The classic 2x2 DiD is preserved when we collapse the rollout
into a simple before/after by-treatment-status comparison.

Run:  python nomnom/episodes/did_city_rollout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nomnom.dgp import sample  # noqa: E402


def main() -> None:
    print("=" * 64)
    print("NomNom Episode: DiD — city-by-city notification rollout")
    print("=" * 64)
    rng = np.random.default_rng(42)
    df = sample(30_000, seed=42).drop(columns=["U"])

    # 10 cities, 5 in the rollout group. post=1 means after rollout.
    n = len(df)
    city = rng.integers(0, 10, n)
    rollout_cities = {0, 1, 2, 4, 7}
    group = np.isin(city, list(rollout_cities)).astype(float)  # treated group
    post = rng.choice([0, 1], n, p=[0.5, 0.5])

    # DiD DGP: group=1 cities get +0.05 during post=1. Continuous outcome.
    y_continuous = df["Y"].to_numpy(float) + 0.05 * group * post + rng.normal(0, 0.01, n)
    y_out = y_continuous  # use continuous outcome for clean DiD recovery

    # 2x2 DiD on group × period (not treatment × period — that's the
    # canonical DiD: we compare the rollout group before/after vs control
    # group before/after)
    g1_post = y_out[(group == 1) & (post == 1)].mean()
    g0_post = y_out[(group == 0) & (post == 1)].mean()
    g1_pre = y_out[(group == 1) & (post == 0)].mean()
    g0_pre = y_out[(group == 0) & (post == 0)].mean()

    did = (g1_post - g1_pre) - (g0_post - g0_pre)
    n_each = [((group == g) & (post == p)).sum() for g in (1, 0) for p in (1, 0)]
    print(f"2x2 DiD table:")
    print(f"  rollout group: pre={g1_pre:.4f} (n={n_each[2]}), "
          f"post={g1_post:.4f} (n={n_each[0]}), "
          f"delta={g1_post-g1_pre:+.4f}")
    print(f"  control group: pre={g0_pre:.4f} (n={n_each[3]}), "
          f"post={g0_post:.4f} (n={n_each[1]}), "
          f"delta={g0_post-g0_pre:+.4f}")
    print(f"  DiD estimate : {did:+.4f}  (true DiD effect = +0.05)")
    assert abs(did - 0.05) < 0.03

    # pre-trend check: the two groups should have similar pre-period levels
    pre_gap = g1_pre - g0_pre
    print(f"  pre-trend gap : {pre_gap:+.4f}  "
          f"(should be ~0 under parallel trends)")
    assert abs(pre_gap) < 0.02

    print("\nUCL stations demonstrated:")
    print("  1 (ASSUME): parallel trends — treated cities would have followed")
    print("     the same path as control cities absent the change")
    print("  5 (MODEL): 2x2 DiD on the binary rollout")
    print("  7 (TEST): pre-trends check — the gap before rollout is near zero")


if __name__ == "__main__":
    main()
