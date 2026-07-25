"""Tier-1 Gallery #1 — Berkeley graduate admissions (Bickel, Hammel &
O'Connell 1975, Science; LR section 3).

The canonical Simpson's paradox: aggregate admission rates look biased
against women; department-level rates (the correct causal stratification,
since department is a confounder of gender -> admission via application
choice) mostly favor women.

The 6-department table is the public, canonical dataset from the 1975 paper.

Run:  python notebooks/tier1_gallery/gallery_berkeley_simpson.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# dept: (men_applied, men_admitted, women_applied, women_admitted)
BERKELEY_1973 = {
    "A": (825, 512, 108, 89),
    "B": (560, 353, 25, 17),
    "C": (325, 120, 593, 202),
    "D": (417, 138, 375, 131),
    "E": (191, 53, 393, 94),
    "F": (373, 22, 341, 24),
}


def main() -> None:
    print("=" * 64)
    print("Berkeley graduate admissions 1973 — Simpson's paradox")
    print("=" * 64)
    m_app = sum(v[0] for v in BERKELEY_1973.values())
    m_adm = sum(v[1] for v in BERKELEY_1973.values())
    w_app = sum(v[2] for v in BERKELEY_1973.values())
    w_adm = sum(v[3] for v in BERKELEY_1973.values())
    print(f"AGGREGATE   men {m_adm/m_app:.1%} ({m_adm}/{m_app})   "
          f"women {w_adm/w_app:.1%} ({w_adm}/{w_app})")
    print("Looks like a 14-point gap against women.\n")

    print(f"{'dept':>4}  {'men rate':>8}  {'women rate':>10}  {'women apply share':>17}")
    favor_women = 0
    for d, (ma, mad, wa, wad) in BERKELEY_1973.items():
        mr, wr = mad / ma, wad / wa
        favor_women += wr > mr
        print(f"{d:>4}  {mr:>8.1%}  {wr:>10.1%}  {wa/(ma+wa):>17.1%}")
    print(f"\nDepartments where women's rate >= men's: {favor_women}/6")
    assert m_adm / m_app - w_adm / w_app > 0.10        # aggregate gap exists
    assert favor_women >= 4                            # but reverses within strata
    print("Women applied disproportionately to HARDER departments (C–F have")
    print("low rates for everyone). Department is a CONFOUNDER of the")
    print("gender -> admission relationship: it affects both the mix of")
    print("applicants and the admission rate. Conditioning on it — the causal")
    print("stratification — dissolves the paradox.")
    print("\nUCL note: 'adjust or not?' was decided by the GRAPH (department is")
    print("pre-treatment and affects both variables), not by the data pattern.")


if __name__ == "__main__":
    main()
