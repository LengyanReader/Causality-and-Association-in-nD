"""P4 demo: the event-driven UCL handling a drifting world (plan section 4.1).

Simulated event stream:
  1. static batch #1  -> baseline established
  2. static batch #2  -> monitors run, no alarm
  3. holiday batch    -> drift alarm, localized to M, actuator re-runs the loop

Usage (from repo root, conda env causality-nd):
    python loops/run_event_loop_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json  # noqa: E402

from nomnom.dgp import HOLIDAY, STATIC, ground_truth, sample  # noqa: E402
from loops.event_runner import UCLEventLoop  # noqa: E402


def main() -> int:
    loop = UCLEventLoop()
    truth_h = ground_truth(regime=HOLIDAY, n_mc=200_000, seed=999)

    print("=" * 64)
    print("EVENT-DRIVEN UCL — simulated production event stream")
    print("=" * 64)

    r1 = loop.emit("batch_arrived",
                   df=sample(10_000, regime=STATIC, seed=100).drop(columns=["U"]),
                   tag="static#1", seed=0)
    print(f"event 1 (static#1 ) -> {r1['action']:10s} "
          f"ATE={r1['report'].estimate.estimate:+.4f}")

    r2 = loop.emit("batch_arrived",
                   df=sample(10_000, regime=STATIC, seed=200).drop(columns=["U"]),
                   tag="static#2", seed=1)
    print(f"event 2 (static#2 ) -> {r2['action']:10s} "
          f"drift={r2['evolution']['drift_detected']}")

    r3 = loop.emit("batch_arrived",
                   df=sample(10_000, regime=HOLIDAY, seed=300).drop(columns=["U"]),
                   tag="holiday#1", seed=2)
    print(f"event 3 (holiday#1) -> {r3['action']:10s} "
          f"drift={r3['evolution']['drift_detected']} "
          f"localized={r3['evolution']['localized_mechanism']}")
    rep = r3["report"]
    covers = rep.estimate.ci_low <= truth_h["ate"] <= rep.estimate.ci_high
    print(f"   actuator re-run   -> ATE={rep.estimate.estimate:+.4f} "
          f"(holiday truth {truth_h['ate']:+.4f}, covers={covers}, "
          f"green={rep.tests.all_green})")

    out_path = REPO_ROOT / "runs" / "event_loop_log.json"
    out_path.write_text(json.dumps(
        {"log": loop.log,
         "holiday_truth": truth_h,
         "covers_truth": covers},
        indent=2, default=str))
    print(f"event log written to : {out_path}")

    ok = (r1["action"] == "baseline" and r2["action"] == "monitor_ok"
          and r3["action"] == "rerun"
          and r3["evolution"]["localized_mechanism"] == "M"
          and covers and rep.tests.all_green)
    print(f"LOOP OK: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
