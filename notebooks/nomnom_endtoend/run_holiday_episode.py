"""Holiday drift episode: the UCL meta-loop in action (plan section 5.2, P3).

Scenario: the platform was analyzed in the static regime. A new batch of data
arrives from the holiday season. The EVOLVE station must
  1. detect that the world changed (mechanism-stability monitor),
  2. localize the change to the correct mechanism (T -> M, app-open response),
  3. confirm the causal mechanism of interest (Y | parents) stayed invariant,
  4. fire the actuator: re-run the UCL pass on the new regime and recover the
     holiday ground truth.

Usage (from repo root, conda env causality-nd):
    python notebooks/nomnom_endtoend/run_holiday_episode.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nomnom.dgp import HOLIDAY, STATIC, ground_truth, sample  # noqa: E402
from ucl.engine import run_pass  # noqa: E402
from ucl.stations.evolve import evolve  # noqa: E402


def main() -> int:
    # 1. Baseline: static-regime pass (the world we modeled)
    base_report, _ = run_pass(regime="static", n=20_000, seed=0)
    graph = base_report.graph
    spec = base_report.estimand
    features = base_report.features
    df_ref = sample(10_000, regime=STATIC, seed=100).drop(columns=["U"])

    # 2. New batch arrives: holiday season
    df_new = sample(10_000, regime=HOLIDAY, seed=300).drop(columns=["U"])

    # 3. EVOLVE: monitors + localization
    entries, evolution_report = evolve(graph, spec, features, df_ref, df_new, seed=0)

    print("=" * 64)
    print("HOLIDAY EPISODE — EVOLVE station on a new data batch")
    print("=" * 64)
    print(f"graph version           : {graph.version}")
    print("monitors:")
    for e in entries:
        marker = "ALARM" if e.status == "alarm" else "ok   "
        metric = f"{e.metric:+.4f}" if e.metric is not None else "  —   "
        print(f"  [{marker}] {e.check:42s} {metric}")
    print(f"drift detected          : {evolution_report['drift_detected']}")
    print(f"localized mechanism     : {evolution_report['localized_mechanism']} "
          f"(expected: M — the notification->app-open mechanism)")

    # 4. Actuator: re-run the pass on the new regime
    print("-" * 64)
    print("ACTUATOR: re-running UCL pass on holiday regime...")
    holiday_report, _ = run_pass(regime="holiday", n=20_000, seed=23)
    truth_h = ground_truth(regime=HOLIDAY, n_mc=200_000, seed=999)
    covers = bool(
        holiday_report.estimate.ci_low <= truth_h["ate"] <= holiday_report.estimate.ci_high
    )
    print(f"holiday ATE estimate    : {holiday_report.estimate.estimate:+.4f} "
          f"[{holiday_report.estimate.ci_low:+.4f}, {holiday_report.estimate.ci_high:+.4f}]")
    print(f"holiday ground truth    : {truth_h['ate']:+.4f}")
    print(f"CI covers truth         : {covers}")
    print(f"refuters all green      : {holiday_report.tests.all_green}")

    out = {
        "baseline_static": base_report.to_dict(),
        "evolution": [e.to_dict() for e in entries],
        "evolution_report": evolution_report,
        "holiday_pass": holiday_report.to_dict(),
        "holiday_truth": truth_h,
        "covers_truth": covers,
    }
    out_path = REPO_ROOT / "runs" / "holiday_episode_report.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"report written to       : {out_path}")

    ok = (
        evolution_report["drift_detected"]
        and evolution_report["localized_mechanism"] == "M"
        and covers
        and holiday_report.tests.all_green
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
