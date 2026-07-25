"""Environment regimes for NomNom (plan §5.2: drift / transportability episode).

The holiday regime changes *exactly one mechanism*: the notification → app-open
equation (users habituate to notifications) plus a seasonal shift in rain.
This is what the EVOLVE station's invariance monitor must detect and localize.
"""

from nomnom.dgp import STATIC, HOLIDAY, Regime

REGIMES: dict[str, Regime] = {r.name: r for r in (STATIC, HOLIDAY)}


def get(name: str) -> Regime:
    if name not in REGIMES:
        raise KeyError(f"Unknown regime {name!r}; available: {sorted(REGIMES)}")
    return REGIMES[name]
