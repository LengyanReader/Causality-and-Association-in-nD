"""Graph utilities: identification compiled from the AssumptionGraph (plan P2).

Implements the back-door criterion by *graph surgery*: in the back-door graph
(edges out of the treatment deleted), Z satisfies the back-door criterion iff
T ⊥ Y | Z (Pearl 1995; LR §4). d-separation is delegated to networkx.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx

from ucl.contracts.artifacts import AssumptionGraph

try:  # networkx >= 3.3
    from networkx.algorithms.d_separation import is_d_separator

    _HAS_DSEP = True
except ImportError:  # pragma: no cover
    _HAS_DSEP = False


def to_nx(graph: AssumptionGraph) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(set(graph.observed) | set(graph.latent))
    g.add_edges_from(graph.edges)
    return g


def _d_separated(g: nx.DiGraph, x: set[str], y: set[str], z: set[str]) -> bool:
    if _HAS_DSEP:
        return bool(is_d_separator(g, x, y, z))
    raise RuntimeError(
        "networkx.algorithms.d_separation is required (networkx >= 3.3)."
    )


def descendants(g: nx.DiGraph, node: str) -> set[str]:
    return nx.descendants(g, node)


def backdoor_graph(g: nx.DiGraph, treatment: str) -> nx.DiGraph:
    """Delete all edges *out of* the treatment (Pearl's back-door graph)."""
    bg = g.copy()
    bg.remove_edges_from(list(bg.out_edges(treatment)))
    return bg


def satisfies_backdoor(
    graph: AssumptionGraph,
    treatment: str,
    outcome: str,
    z: set[str],
) -> bool:
    """Check the back-door criterion for a candidate adjustment set Z.

    Conditions (Pearl 2009, Def. 3.3.1):
      1. No node in Z is a descendant of treatment.
      2. Z blocks every path between treatment and outcome that has an arrow
         into treatment — i.e., d-separation in the back-door graph.
    """
    g = to_nx(graph)
    if z & descendants(g, treatment):
        return False
    bg = backdoor_graph(g, treatment)
    return _d_separated(bg, {treatment}, {outcome}, set(z))


def find_adjustment_set(
    graph: AssumptionGraph,
    treatment: str,
    outcome: str,
    latent_ok: bool = False,
) -> list[str] | None:
    """Smallest observed back-door adjustment set, or None if not identifiable.

    Searches subsets of observed variables in increasing size. The DAGs handled
    here are small; exhaustive search is fine and keeps the criterion exact.
    """
    g = to_nx(graph)
    candidates = sorted(
        v
        for v in graph.observed
        if v not in {treatment, outcome}
        and v not in descendants(g, treatment)  # never adjust for descendants
    )
    for k in range(len(candidates) + 1):
        for combo in combinations(candidates, k):
            if satisfies_backdoor(graph, treatment, outcome, set(combo)):
                return list(combo)
    return None


def on_causal_paths(
    graph: AssumptionGraph, treatment: str, outcome: str
) -> set[str]:
    """Mediators: observed nodes lying on a directed treatment→outcome path."""
    g = to_nx(graph)
    mediators = set()
    for path in nx.all_simple_paths(g, treatment, outcome):
        mediators.update(path[1:-1])
    return mediators & set(graph.observed)


def colliders_of(
    graph: AssumptionGraph, treatment: str, outcome: str
) -> set[str]:
    """Observed common effects of both treatment and outcome (never adjust)."""
    g = to_nx(graph)
    return set(g.successors(treatment)) & set(g.successors(outcome)) & set(graph.observed)
