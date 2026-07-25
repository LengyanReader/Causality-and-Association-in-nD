"""The UCL engine: one full pass through stations 0–7 (plan §2).

Station 8 (EVOLVE) ships in Phase 3; the engine already records a stub
EvolutionLogEntry so the artifact chain is complete.
"""

from __future__ import annotations

from ucl.contracts.artifacts import EvolutionLogEntry, UCLRunReport
from ucl.stations import (
    assume,
    compile_features,
    evaluate,
    frame,
    identify,
    load_data,
    model,
    test_suite,
)


def run_pass(
    regime: str = "static",
    n: int = 20_000,
    seed: int = 0,
) -> tuple[UCLRunReport, list[EvolutionLogEntry]]:
    # Station 0 — FRAME
    spec = frame()
    # Station 1 — ASSUME
    graph = assume()
    # Station 2 — IDENTIFY
    proof = identify(graph, spec)
    if not proof.identified:
        raise RuntimeError(
            f"Estimand {spec.name!r} is not identified under graph v{graph.version}; "
            "loop must route to discovery or redesign (actuator, station 2)."
        )
    # Station 3 — DATA
    df, data_contract = load_data(proof, regime_name=regime, n=n, seed=seed)
    # Station 4 — FEATURE
    features = compile_features(graph, proof)
    # Station 5 — MODEL
    bundle = model(df, spec, features, seed=seed)
    # Station 6 — EVALUATE
    evaluation = evaluate(df, spec, features, bundle)
    # Station 7 — TEST
    suite = test_suite(df, spec, features, evaluation, graph, seed=seed)
    # Station 8 — EVOLVE (stub until Phase 3)
    evolution = [
        EvolutionLogEntry(
            check="evolve_stub",
            status="ok",
            detail={"note": "invariance/drift monitors land in Phase 3"},
            graph_version=graph.version,
        )
    ]

    # Loop invariant 1: every artifact carries the same graph version
    versions = {
        proof.graph_version,
        data_contract.graph_version,
        features.graph_version,
        bundle.graph_version,
        evaluation.graph_version,
        suite.graph_version,
    }
    assert versions == {graph.version}, f"invariant 1 violated: {versions}"

    report = UCLRunReport(
        estimand=spec,
        graph=graph,
        identification=proof,
        data=data_contract,
        features=features,
        estimate=bundle,
        evaluation=evaluation,
        tests=suite,
    )
    return report, evolution
