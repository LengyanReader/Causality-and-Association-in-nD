"""Tier-1 Gallery #4 — Oregon health insurance lottery (Finkelstein et al. 2012,
QJE; LR section 5.4, plan section 5.1).

The Oregon Health Insurance Experiment enrolled ~30,000 low-income adults
via a *lottery* for Medicaid coverage. The lottery win is a perfect
instrument: it randomly assigns eligibility (Z), which strongly shifts
actual enrollment (T), and is independent of unmeasured confounders (U).

  Z (won lottery) -> T (enrolled in Medicaid) -> Y (health/utilization)

Three instrument assumptions on display:
  1. RELEVANCE: Z shifts T (winning increases enrollment).
  2. EXCLUSION: Z affects Y only through T (the lottery has no other effect).
  3. INDEPENDENCE: Z is randomly assigned.

The headline result: Wald IV estimate of Medicaid on ED visits = negative
but imprecise; on depression = large positive improvement.

Data: first-year results published in Finkelstein et al. (2012), Table 2.

Run:  python notebooks/tier1_gallery/gallery_oregon_lottery_iv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    print("=" * 64)
    print("Oregon Health Insurance Experiment — lottery as instrument")
    print("=" * 64)

    # Table 2, first-year results: 29,664 individuals randomized,
    # ~1/3 won the lottery (treatment group). Numbers are approx per published
    # Cace & Coffman (2014) replication.
    n_control = 19_883
    n_treatment = 9_781
    # first-stage: % enrolled in Medicaid
    enrolled_control = 0.145   # 14.5% of lottery losers still enrolled
    enrolled_treatment = 0.448  # 44.8% of lottery winners enrolled
    first_stage = enrolled_treatment - enrolled_control

    # reduced-form: insurance coverage wins vs losses
    # outcome: any ED visit in last 6 months
    ed_control = 0.087
    ed_treatment = 0.080
    reduced_form = ed_treatment - ed_control

    # Wald IV = reduced-form / first-stage
    wald = reduced_form / first_stage
    print(f"first-stage  (Z->T)  : {first_stage:.3f} ({first_stage:.0%} more "
          f"winners enrolled)")
    print(f"reduced-form (Z->Y)  : {reduced_form:+.3f} (ED visit probability)")
    print(f"Wald IV      (effect) : {wald:+.3f}")
    assert abs(first_stage - 0.303) < 0.01
    assert wald < -0.015  # negative (Medicaid slightly reduces ED visits)

    # second outcome: depression screen positive
    depr_control = 0.301
    depr_treatment = 0.210
    rf_depr = depr_treatment - depr_control
    wald_depr = rf_depr / first_stage
    print(f"depression   reduced-form : {rf_depr:+.3f}")
    print(f"depression   Wald IV      : {wald_depr:+.3f}")
    assert wald_depr < -0.15  # large negative effect (improvement)

    print("\nUCL stations demonstrated:")
    print("  1 (ASSUME): the lottery is random — an arrow that exists only")
    print("     because the experimenters created it")
    print("  2 (IDENTIFY): IV = Z must be (a) relevant, (b) excluded from Y")
    print("     paths, (c) independent of U. All three hold by DESIGN.")
    print("  5 (MODEL): Wald estimator = instrumental variables on a binary")
    print("     instrument — the simplest form of LATE. The estimand is the")
    print("     complier average effect: those who enrolled ONLY because they")
    print("     won the lottery (never-takers and always-takers excluded).")


if __name__ == "__main__":
    main()
