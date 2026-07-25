"""NomNom's causal graph — the AssumptionGraph for the reference use case.

Encodes the *believed* structure (station ASSUME). In this reference project it
mirrors the true DGP; the EVOLVE station exists precisely for the case where it
doesn't. Edges carry rationale; declared absent edges are the falsifiable part.
"""

from ucl.contracts.artifacts import AssumptionGraph


def nomnom_graph() -> AssumptionGraph:
    edges = [
        # observed confounders -> T and Y
        ("weekend", "T"), ("weekend", "Y"),
        ("rain", "T"), ("rain", "Y"),
        ("payday", "T"), ("payday", "Y"),
        # latent hunger confounds Y and M; the platform targets on the measured
        # proxy W (U -> W -> T), so W blocks the confounding path
        ("U", "W"), ("W", "T"), ("U", "Y"), ("U", "M"), ("U", "NC"),
        # instrument: randomized send-time jitter affects only T
        ("Z", "T"),
        # causal chain of interest
        ("T", "M"), ("M", "Y"), ("T", "Y"),
        # loyalty -> coupon (RDD) -> Y
        ("loyalty", "coupon"), ("coupon", "Y"),
        # collider: engagement score is a common effect of T and Y
        ("T", "S"), ("Y", "S"),
        # segment modifies the T -> Y effect (effect modifier)
        ("segment", "Y"),
    ]
    return AssumptionGraph(
        edges=edges,
        observed=["segment", "weekend", "rain", "payday", "W", "Z", "T", "M",
                  "loyalty", "coupon", "Y", "S", "NC"],
        latent=["U"],
        absent_edges=[
            ("Z", "Y"), ("Z", "U"), ("Z", "M"),   # instrument exclusion restrictions
            ("T", "NC"),                            # negative-control: no T effect on battery
            ("S", "Y"), ("S", "T"),                 # S is a pure effect, never a cause
            ("coupon", "T"), ("loyalty", "T"),      # RDD running variable unrelated to targeting
        ],
        node_roles={
            "T": "treatment", "Y": "outcome", "M": "mediator", "S": "collider",
            "Z": "instrument", "NC": "negative_control_outcome", "W": "proxy_confounder",
            "U": "latent_confounder", "coupon": "rdd_assignment", "loyalty": "rdd_running",
        },
        rationale={
            "W->T": "platform targets notifications using measured app-use history (hunger proxy)",
            "U->T (absent)": "true hunger is unobserved; targeting can only use the proxy W",
            "U->Y": "hunger directly drives orders",
            "Z->T": "send-time jitter is randomized by the experiment platform",
            "T->S,Y->S": "engagement score is computed from notifications and orders",
            "T->NC (absent)": "push notifications cannot drain battery; appetite does correlate",
        },
    )
