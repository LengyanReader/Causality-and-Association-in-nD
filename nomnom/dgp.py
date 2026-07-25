"""NomNom Eats — ground-truth data-generating process (plan §5.2/§5.3).

A food-delivery platform. Business question: do push notifications (T) cause
orders (Y), and for whom?

Structural equations (static regime; holiday regime modifies M's equation):

  segment      ~ Bernoulli(0.6)                      # 1 = loyal, 0 = new
  weekend      ~ Bernoulli(0.28)
  rain         ~ Bernoulli(0.30)
  payday       ~ Bernoulli(0.14)
  hunger U     ~ N(0, 1)                             # LATENT confounder
  app_use W    = U + N(0, 0.5)                       # measured proxy of U
  jitter Z     ~ Bernoulli(0.5)                      # randomized send-time (INSTRUMENT)
  notify T     ~ Bernoulli(σ(-0.6 + 0.9·W + 0.5·wknd + 0.3·rain + 0.4·payday + 1.1Z))
                 # the platform targets on *measured* app-use W, not true hunger U
                 # (U → W → T), so {W, weekend, rain, payday} satisfies back-door
  open M       ~ Bernoulli(σ(-1.2 + β_TM·T + 0.5U))  # MEDIATOR (β_TM = 1.6 static)
  loyalty L    ~ N(480, 60)
  coupon D     = 1[L ≥ 500]                          # RDD assignment (sharp)
  order Y      ~ Bernoulli(σ(-1.3 + 0.5U + 0.55·wknd + 0.35·rain + 0.3·payday
                              + 0.9·M + τ_seg·T + 0.5·D))
                 τ_new = 0.4, τ_loyal = 1.1 (logit-scale direct effect)
  engage S     ~ Bernoulli(σ(-0.5 + 0.9·T + 1.2·Y))  # COLLIDER — never adjust
  battery NC   ~ Bernoulli(σ(-1.0 + 0.7·U))          # NEGATIVE-CONTROL outcome:
                                                     # shares confounder U, no T effect

Ground truth is computed by Monte Carlo on the *same* exogenous draws under
do(T=1) vs do(T=0) — see `ground_truth`. All downstream estimators are
unit-tested against it (causal_ci/).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class Regime:
    """Environment parameters; the holiday regime flips exactly one mechanism."""

    name: str = "static"
    beta_tm: float = 1.6      # T -> M coefficient (notification → app open)
    base_open: float = -1.2   # intercept of the M equation
    p_rain: float = 0.30


STATIC = Regime()
HOLIDAY = Regime(name="holiday", beta_tm=0.4, base_open=-0.6, p_rain=0.15)


@dataclass(frozen=True)
class DGPParams:
    p_loyal: float = 0.6
    p_weekend: float = 0.28
    p_payday: float = 0.14
    p_jitter: float = 0.5
    tau_new: float = 0.4      # direct logit effect of T, new users
    tau_loyal: float = 1.1    # direct logit effect of T, loyal users
    beta_um: float = 0.5      # U -> M
    beta_uy: float = 0.5      # U -> Y
    beta_my: float = 0.9      # M -> Y
    beta_coupon: float = 0.5  # coupon -> Y
    rdd_cutoff: float = 500.0
    loyalty_mu: float = 480.0
    loyalty_sd: float = 60.0


DEFAULT_PARAMS = DGPParams()


def _draw_exogenous(n: int, rng: np.random.Generator, regime: Regime, p: DGPParams):
    return {
        "segment": rng.binomial(1, p.p_loyal, n),
        "weekend": rng.binomial(1, p.p_weekend, n),
        "rain": rng.binomial(1, regime.p_rain, n),
        "payday": rng.binomial(1, p.p_payday, n),
        "U": rng.normal(0.0, 1.0, n),
        "eps_w": rng.normal(0.0, 0.5, n),
        "Z": rng.binomial(1, p.p_jitter, n),
        "u_t": rng.uniform(size=n),
        "u_m": rng.uniform(size=n),
        "loyalty": rng.normal(p.loyalty_mu, p.loyalty_sd, n),
        "u_y": rng.uniform(size=n),
        "u_s": rng.uniform(size=n),
        "u_nc": rng.uniform(size=n),
    }


def _structural(exo: dict, regime: Regime, p: DGPParams, t_value: np.ndarray | None):
    """Evaluate the SCM; t_value=None → observational T, else do(T=t_value)."""
    U = exo["U"]
    W = U + exo["eps_w"]
    Z = exo["Z"]
    if t_value is None:
        logit_t = (
            -0.6 + 0.9 * W + 0.5 * exo["weekend"] + 0.3 * exo["rain"]
            + 0.4 * exo["payday"] + 1.1 * exo["Z"]
        )
        T = (_sigmoid(logit_t) > exo["u_t"]).astype(int)
    else:
        T = np.full_like(U, t_value, dtype=int)
    M = (_sigmoid(regime.base_open + regime.beta_tm * T + p.beta_um * U) > exo["u_m"]).astype(int)
    coupon = (exo["loyalty"] >= p.rdd_cutoff).astype(int)
    tau = np.where(exo["segment"] == 1, p.tau_loyal, p.tau_new)
    logit_y = (
        -1.3 + p.beta_uy * U + 0.55 * exo["weekend"] + 0.35 * exo["rain"]
        + 0.3 * exo["payday"] + p.beta_my * M + tau * T + p.beta_coupon * coupon
    )
    Y = (_sigmoid(logit_y) > exo["u_y"]).astype(int)
    S = (_sigmoid(-0.5 + 0.9 * T + 1.2 * Y) > exo["u_s"]).astype(int)
    NC = (_sigmoid(-1.0 + 0.7 * U) > exo["u_nc"]).astype(int)
    return pd.DataFrame(
        {
            "segment": exo["segment"],
            "weekend": exo["weekend"],
            "rain": exo["rain"],
            "payday": exo["payday"],
            "U": U,          # latent — shipped for testing only, marked unobserved
            "W": W,
            "Z": Z,
            "T": T,
            "M": M,
            "loyalty": exo["loyalty"],
            "coupon": coupon,
            "Y": Y,
            "S": S,
            "NC": NC,
        }
    )


def sample(
    n: int,
    regime: Regime = STATIC,
    params: DGPParams = DEFAULT_PARAMS,
    seed: int = 0,
) -> pd.DataFrame:
    """Observational sample (T assigned by the platform's targeting policy)."""
    rng = np.random.default_rng(seed)
    exo = _draw_exogenous(n, rng, regime, params)
    return _structural(exo, regime, params, t_value=None)


def ground_truth(
    regime: Regime = STATIC,
    params: DGPParams = DEFAULT_PARAMS,
    n_mc: int = 400_000,
    seed: int = 10_000,
) -> dict:
    """Exact (Monte-Carlo) causal effects under do(T), with common random numbers."""
    rng = np.random.default_rng(seed)
    exo = _draw_exogenous(n_mc, rng, regime, params)  # common random numbers
    obs = _structural(exo, regime, params, t_value=None)
    y1 = _structural(exo, regime, params, t_value=np.ones(1, dtype=int))["Y"].to_numpy()
    y0 = _structural(exo, regime, params, t_value=np.zeros(1, dtype=int))["Y"].to_numpy()
    seg = obs["segment"].to_numpy()
    ite = y1 - y0
    return {
        "ate": float(ite.mean()),
        "mu1": float(y1.mean()),
        "mu0": float(y0.mean()),
        "cate_loyal": float(ite[seg == 1].mean()),
        "cate_new": float(ite[seg == 0].mean()),
        "regime": regime.name,
        "n_mc": n_mc,
    }
