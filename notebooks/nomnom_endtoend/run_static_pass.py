"""End-to-end UCL pass on the static NomNom regime (plan §8, Phase 1).

Runs stations 0–7, compares the estimate against ground truth, writes the full
artifact chain to runs/static_pass_report.json, and prints a summary.

Usage (from repo root, conda env causality-nd):
    python notebooks/nomnom_endtoend/run_static_pass.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nomnom.dgp import ground_truth  # noqa: E402
from ucl.engine import run_pass  # noqa: E402


def main() -> int:
    report, evolution = run_pass(regime="static", n=20_000, seed=0)
    truth = ground_truth()

    out = {
        "ucl_report": report.to_dict(),
        "evolution": [e.to_dict() for e in evolution],
        "ground_truth": truth,
        "estimate_vs_truth": {
            "estimate": report.estimate.estimate,
            "ci": [report.estimate.ci_low, report.estimate.ci_high],
            "truth_ate": truth["ate"],
            "covers_truth": bool(
                report.estimate.ci_low <= truth["ate"] <= report.estimate.ci_high
            ),
        },
    }
    out_path = REPO_ROOT / "runs" / "static_pass_report.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 64)
    print("UCL static pass — NomNom Eats: do push notifications cause orders?")
    print("=" * 64)
    print(f"graph version          : {report.graph.version}")
    print(f"identification         : {report.identification.criterion} "
          f"via {report.identification.adjustment_set}")
    print(f"excluded features      : {report.features.excluded}")
    print(f"positivity ok          : {report.data.positivity_ok}")
    print(f"ATE estimate           : {report.estimate.estimate:+.4f} "
          f"[{report.estimate.ci_low:+.4f}, {report.estimate.ci_high:+.4f}]")
    print(f"ground-truth ATE       : {truth['ate']:+.4f} "
          f"(new {truth['cate_new']:+.4f} / loyal {truth['cate_loyal']:+.4f})")
    print(f"CI covers truth        : {out['estimate_vs_truth']['covers_truth']}")
    print(f"E-value                : {report.evaluation.e_value:.2f} "
          f"(RR {report.evaluation.risk_ratio:.2f})")
    print(f"max |SMD| after weight : {report.evaluation.balance['max_abs_smd']:.3f}")
    print("refuters:")
    for r in report.tests.refuters:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
    print("invariant checks:")
    for r in report.tests.invariant_checks:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}")
    print(f"ALL GREEN              : {report.tests.all_green}")
    print(f"report written to      : {out_path}")
    return 0 if report.tests.all_green else 1


if __name__ == "__main__":
    raise SystemExit(main())
