"""NHEFS data loader — the real NHANES I Epidemiologic Follow-up Study.

The dataset analyzed in Hernan & Robins (2020), *Causal Inference: What If*,
the canonical textbook of modern causal inference.

Question: Does quitting smoking (qsmk) cause weight gain (wt82_71)?

  - 1,566 Americans from NHANES I (1971-1975), followed up in 1982
  - Treatment: qsmk = 1 if the person quit smoking between baseline and follow-up
  - Outcome: wt82_71 = weight change (kg) between 1971 and 1982
  - 18 covariates including age, sex, race, smoking intensity, exercise,
    education, and baseline weight

Uses the `causallib` (IBM) package to load the cleaned dataset.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_nhefs(force_download: bool = False) -> pd.DataFrame:
    """Load the NHEFS dataset, caching to notebooks/data/."""
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = DATA_DIR / "nhefs.csv"

    if cache_path.exists() and not force_download:
        return pd.read_csv(cache_path)

    # Try multiple sources for the data
    df = None

    # (1) `causallib` package
    try:
        from causallib.datasets import load_nhefs as _load
        data = _load()
        df = pd.concat([data.X, data.a, data.y], axis=1)
    except Exception:
        pass

    # (2) fallback: direct download of Hernan & Robins' CSV
    if df is None:
        import ssl
        import urllib.request

        url = (
            "https://cdn1.sph.harvard.edu/wp-content/uploads/"
            "sites/1268/1268/20/nhefs.csv"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
                df = pd.read_csv(r)
        except Exception:
            pass

    # (3) last fallback: bundled sample (shipped with this repo)
    if df is None:
        sample_path = DATA_DIR / "nhefs_sample.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path)

    if df is None:
        raise RuntimeError(
            "Could not load NHEFS data. Install causallib: "
            "pip install causallib"
        )

    df.to_csv(cache_path, index=False)
    return df


def main():
    df = load_nhefs()
    print(f"NHEFS loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    qsmk = df.get("qsmk", None)
    if qsmk is not None:
        print(f"Quit smoking (qsmk=1): {qsmk.sum()} ({qsmk.mean():.1%})")
    wt = df.get("wt82_71", None)
    if wt is not None:
        print(f"Weight change (kg): mean={wt.mean():.2f}, "
              f"sd={wt.std():.2f}")
    print(f"Covariates: {list(df.columns)}")


if __name__ == "__main__":
    main()
