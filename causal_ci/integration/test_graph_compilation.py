"""Integration tests: identification <-> estimation consistency, compiled from
the graph (plan section 4.3, integration layer).

The key property: the test suite is *generated* from the AssumptionGraph, so
editing the graph automatically changes what is estimated and what is tested.
"""

import networkx as nx

from nomnom.dgp import ground_truth, sample
from nomnom.graph import nomnom_graph
from ucl import graph_utils
from ucl.stations import compile_features, frame, identify
from ucl.stations.analysis import aipw_crossfit

TRUTH = ground_truth(n_mc=200_000, seed=999)


def _graph_without(edge):
    g = nomnom_graph()
    g.edges = [e for e in g.edges if e != edge]
    return type(g)(
        edges=g.edges,
        observed=g.observed,
        latent=g.latent,
        absent_edges=g.absent_edges + [edge],
        node_roles=g.node_roles,
        rationale=g.rationale,
    )


def test_every_identified_estimand_has_an_estimator_path():
    """Consistency: identified -> adjustment set -> features -> estimator input."""
    graph = nomnom_graph()
    spec = frame()
    proof = identify(graph, spec)
    assert proof.identified
    features = compile_features(graph, proof)
    df = sample(2_000, seed=11).drop(columns=["U"])
    # every adjustment variable must exist in the data and be non-degenerate
    for v in features.adjustment_set:
        assert v in df.columns and df[v].nunique() > 1
    # every excluded variable must be a collider or mediator per the graph
    colliders = graph_utils.colliders_of(graph, "T", "Y")
    mediators = graph_utils.on_causal_paths(graph, "T", "Y")
    for v in features.excluded:
        assert v in colliders | mediators


def test_deleting_backdoor_edge_changes_adjustment_set():
    """Graph edit: remove W->T (declare targeting ignores app history).

    The back-door search must then drop W from the adjustment set —
    the compile-once-test-everywhere property (plan section 4.2).
    """
    broken_graph = _graph_without(("W", "T"))
    proof = identify(broken_graph, frame())
    assert proof.identified
    assert "W" not in proof.adjustment_set
    assert set(proof.adjustment_set) == {"weekend", "rain", "payday"}
    # graph version must change (provenance)
    assert broken_graph.version != nomnom_graph().version


def test_stale_graph_turns_groundtruth_test_red():
    """The misspecified graph from above produces a biased estimate on data
    that still obeys the TRUE dgp — the pinned-bias test goes red (plan P2
    acceptance)."""
    broken_graph = _graph_without(("W", "T"))
    proof = identify(broken_graph, frame())
    df = sample(30_000, seed=12).drop(columns=["U"])
    res = aipw_crossfit(df, "T", "Y", proof.adjustment_set, seed=12)
    assert abs(res["ate"] - TRUTH["ate"]) > 3 * res["se"]  # red: misses truth


def test_graph_version_flows_through_compilation():
    graph = nomnom_graph()
    proof = identify(graph, frame())
    features = compile_features(graph, proof)
    assert proof.graph_version == graph.version == features.graph_version


def test_instrument_never_enters_adjustment():
    """Z is a cause of T only; adjusting for it would amplify bias (bias-amplifying
    variable). The compiler must keep it out for any graph variant."""
    graph = nomnom_graph()
    proof = identify(graph, frame())
    assert "Z" not in proof.adjustment_set
    # and Z reaches Y only *through* T (exclusion restriction, graph form):
    # removing T must disconnect Z from Y entirely
    g = graph_utils.to_nx(graph)
    assert nx.has_path(g, "Z", "T")
    g_no_t = g.copy()
    g_no_t.remove_node("T")
    assert not nx.has_path(g_no_t, "Z", "Y")
