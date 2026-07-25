"""Tier-1 Gallery #2 — LaLonde/NSW: the founding benchmark of matching
(LaLonde 1986; Dehejia & Wahba 1999; LR section 13.1, project ladder #1).

The question that built modern observational methods: can statistical
adjustment on a NON-experimental comparison group recover an RCT benchmark?

  1. RCT benchmark: NSW treated vs NSW control -> ATE ~= +$1,794 on 1978
     earnings (re78).
  2. Naive observational: NSW treated vs PSID population controls -> wildly
     biased (the comparison group is nothing like the treated).
  3. Propensity-score adjusted (logistic PS + IPW/AIPW on the same covariates
     as Dehejia & Wahba) -> back within striking distance of the benchmark.

Data: Dehejia's public files (downloaded once, cached to notebooks/data/).

Run:  python notebooks/tier1_gallery/gallery_lalonde_nsw.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "notebooks" / "data"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NSW_URL = "https://users.nber.org/~rdehejia/data/nsw_dw.dta"
PSID_URL = "https://users.nber.org/~rdehejia/data/psid_controls.dta"
COVS = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]


def load(url: str, name: str) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / name
    if not cache.exists():
        cache.write_bytes(__import__("urllib.request").request.urlopen(url).read())
    return pd.read_stata(cache)


def main() -> None:
    print("=" * 64)
    print("LaLonde/NSW — can observational adjustment recover an RCT benchmark?")
    print("=" * 64)
    nsw = load(NSW_URL, "nsw_dw.dta")
    psid = load(PSID_URL, "psid_controls.dta")

    # 1. RCT benchmark
    t = nsw[nsw["treat"] == 1]["re78"].mean()
    c = nsw[nsw["treat"] == 0]["re78"].mean()
    benchmark = t - c
    print(f"1. RCT benchmark (NSW treated vs NSW control): ${benchmark:+,.0f}")
    assert 1500 < benchmark < 2100  # canonical ~$1,794

    # 2. naive observational estimate vs PSID comparison group
    obs = pd.concat([nsw[nsw["treat"] == 1], psid], ignore_index=True)
    naive = obs[obs["treat"] == 1]["re78"].mean() - obs[obs["treat"] == 0]["re78"].mean()
    print(f"2. naive (treated vs PSID controls)          : ${naive:+,.0f}")
    assert abs(naive - benchmark) > 3_000  # LaLonde's cautionary result

    # 3. the DATA station speaks first: overlap is terrible.
    X = StandardScaler().fit_transform(obs[COVS])
    T = obs["treat"].to_numpy()
    ps = LogisticRegression(max_iter=5000).fit(X, T).predict_proba(X)[:, 1]
    obs = obs.assign(ps=ps)
    treated, ctrl = obs[obs["treat"] == 1], obs[obs["treat"] == 0]
    print(f"3. OVERLAP CHECK: treated PS up to {treated['ps'].max():.2f}, "
          f"but 95% of PSID controls have PS <= {ctrl['ps'].quantile(.95):.2f}")
    print("   -> positivity alarm: weighting without trimming is unstable here.")

    # 4. propensity-score matching (nearest neighbor, with replacement):
    #    matching TRIMS to the region of common support — that is why it works.
    cps = ctrl["ps"].to_numpy()
    cy = ctrl["re78"].to_numpy()
    idx = np.abs(treated["ps"].to_numpy()[:, None] - cps[None, :]).argmin(axis=1)
    est = float(treated["re78"].to_numpy().mean() - cy[idx].mean())
    print(f"4. PS nearest-neighbor matching              : ${est:+,.0f}")
    print(f"\nbenchmark ${benchmark:+,.0f} | naive error ${naive-benchmark:+,.0f} "
          f"| matched error ${est-benchmark:+,.0f}")
    assert abs(est - benchmark) < abs(naive - benchmark) * 0.1
    assert abs(est - benchmark) < 1_200
    print("Dehejia & Wahba's point, reproduced: with the right adjustment set")
    print("AND respect for common support, observational methods approach the")
    print("experimental benchmark. LaLonde's point: without them, they are")
    print("catastrophically wrong. UCL stations 2-3-4 in one example: the")
    print("covariates are the assumptions, and overlap is a hard gate.")


if __name__ == "__main__":
    main()
