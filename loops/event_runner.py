"""P4 — Event-driven UCL runner (plan section 4.1, loop-engineering level L3).

The Universal Causal Loop as a long-running service:

- `batch_arrived` events trigger either baseline establishment (first batch)
  or an EVOLVE monitor pass against the reference batch;
- drift alarms fire the ACTUATOR: a full re-estimation on the new batch,
  no human in the loop;
- every action is logged to the EvolutionLog for the hill-climbing loop (L4).

The regime label is used ONLY to simulate data arrival — drift decisions come
from the monitors, never from the label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ucl.contracts.artifacts import (
    AssumptionGraph,
    DataContract,
    EstimandSpec,
    UCLRunReport,
)
from ucl.stations import (
    assume,
    compile_features,
    evaluate,
    frame,
    identify,
    model,
    test_suite,
)
from ucl.stations.evolve import evolve


@dataclass
class UCLEventLoop:
    """A standing UCL instance with its graph, identification, and reference batch."""

    spec: EstimandSpec = field(default_factory=frame)
    graph: AssumptionGraph = field(default_factory=assume)

    def __post_init__(self) -> None:
        self.proof = identify(self.graph, self.spec)
        if not self.proof.identified:
            raise RuntimeError("estimand not identified under the standing graph")
        self.features = compile_features(self.graph, self.proof)
        self.reference: pd.DataFrame | None = None
        self.log: list[dict[str, Any]] = []
        self.reports: dict[str, UCLRunReport] = {}

    # -- public API ------------------------------------------------------
    def emit(self, event_type: str, **payload) -> dict[str, Any]:
        handler = getattr(self, f"_on_{event_type}", None)
        if handler is None:
            raise ValueError(f"no handler for event type {event_type!r}")
        return handler(**payload)

    # -- handlers --------------------------------------------------------
    def _on_batch_arrived(self, df: pd.DataFrame, tag: str, seed: int = 0) -> dict[str, Any]:
        if self.reference is None:
            report = self._estimate(df, tag, seed)
            self.reference = df
            self.log.append({"event": "baseline_established", "tag": tag,
                             "ate": report.estimate.estimate})
            return {"action": "baseline", "report": report}

        entries, evo = evolve(self.graph, self.spec, self.features, self.reference, df, seed=seed)
        rec = {"event": "batch_arrived", "tag": tag,
               "drift_detected": evo["drift_detected"],
               "localized_mechanism": evo["localized_mechanism"],
               "n_alarms": evo["n_alarms"]}
        self.log.append(rec)
        if evo["drift_detected"]:
            report = self._estimate(df, tag, seed)
            self.log.append({
                "event": "actuator_rerun", "tag": tag,
                "reason": f"mechanism drift localized to {evo['localized_mechanism']}",
                "ate": report.estimate.estimate,
                "all_green": report.tests.all_green,
            })
            return {"action": "rerun", "report": report, "evolution": evo,
                    "entries": entries}
        return {"action": "monitor_ok", "evolution": evo, "entries": entries}

    # -- internals -------------------------------------------------------
    def _data_contract(self, df: pd.DataFrame, tag: str) -> DataContract:
        X = df[self.proof.adjustment_set].to_numpy(float)
        ps = LogisticRegression(max_iter=1000).fit(X, df["T"]).predict_proba(X)[:, 1]
        return DataContract(
            source=f"event_batch:{tag}",
            n_rows=len(df),
            columns=list(df.columns),
            regime=tag,
            overlap={"ps_min": float(ps.min()), "ps_max": float(ps.max())},
            positivity_ok=bool(0.01 <= ps.min() and ps.max() <= 0.99),
            graph_version=self.graph.version,
        )

    def _estimate(self, df: pd.DataFrame, tag: str, seed: int) -> UCLRunReport:
        bundle = model(df, self.spec, self.features, seed=seed)
        evaluation = evaluate(df, self.spec, self.features, bundle)
        suite = test_suite(df, self.spec, self.features, evaluation, self.graph, seed=seed)
        report = UCLRunReport(
            estimand=self.spec,
            graph=self.graph,
            identification=self.proof,
            data=self._data_contract(df, tag),
            features=self.features,
            estimate=bundle,
            evaluation=evaluation,
            tests=suite,
        )
        self.reports[tag] = report
        return report

    def dump_log(self) -> str:
        return json.dumps(self.log, indent=2, default=str)
