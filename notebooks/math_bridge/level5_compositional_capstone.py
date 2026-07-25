"""Math Bridge — Level 5: compositional capstone (plan section 6, Level 5).

The do-operator as DIAGRAM SURGERY, in the spirit of Markov categories and
string-diagram causality (Fritz 2020; Jacobs, Kissinger & Zanasi 2019;
Fong 2013 — LR section 14.2).

A structural causal model is a wiring diagram of mechanisms (boxes) joined by
shared variables (wires), with two structural maps available for free:
COPY (a variable may feed several boxes — the source of confounding) and
DISCARD (a wire may be summed out). Interventional calculus is one move:
to do(T=t), CUT the input wire of the T-box and clamp it to t.

With a tiny exact tensor engine we show, with exact numbers:
  1. conditioning (P(Y|T)) != surgery (P(Y|do(T)))
  2. back-door adjustment == surgery — the criterion is a *theorem about the
     diagram*, re-derived rather than axiomatized
  3. the copy map is what makes confounding *visible*

Run:  python notebooks/math_bridge/level5_compositional_capstone.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ----------------------------------------------------------------------------
# A minimal factor engine ("boxes and wires"). A Factor is a joint/conditional
# table with named axes; composition = tensor product over shared wires
# (broadcasting); discard = marginalize; condition = slice + renormalize.
# ----------------------------------------------------------------------------

class Factor:
    def __init__(self, vars_: tuple[str, ...], table):
        table = np.asarray(table, dtype=float)
        assert len(vars_) == table.ndim
        self.vars = tuple(vars_)
        self.table = table

    def __mul__(self, other: "Factor") -> "Factor":
        all_vars = tuple(dict.fromkeys(self.vars + other.vars))

        def expand(f: "Factor") -> np.ndarray:
            missing = [v for v in all_vars if v not in f.vars]
            t = f.table.reshape(list(f.table.shape) + [1] * len(missing))
            order = [(list(f.vars) + missing).index(v) for v in all_vars]
            return np.transpose(t, order)

        return Factor(all_vars, expand(self) * expand(other))

    def marginalize(self, var: str) -> "Factor":
        i = self.vars.index(var)
        return Factor(tuple(v for v in self.vars if v != var), self.table.sum(axis=i))

    def condition(self, var: str, value: int) -> "Factor":
        """Slice a wire at `value` and renormalize — Bayesian conditioning."""
        i = self.vars.index(var)
        t = np.take(self.table, value, axis=i)
        rest = tuple(v for v in self.vars if v != var)
        return Factor(rest, t / t.sum())

    def value(self, var: str) -> float:
        """P(var=1) of a normalized single-variable factor."""
        assert self.vars == (var,)
        return float(self.table[1])


# ----------------------------------------------------------------------------
# NomNom-in-miniature: binary U -> W -> T -> M -> Y, plus U -> Y.
# COPY(U) feeds both the W-box and the Y-box: the visible source of confounding.
# ----------------------------------------------------------------------------
P_U = 0.5
P_W_given_U = [0.2, 0.8]   # P(W=1 | U=0/1)
P_T_given_W = [0.1, 0.9]   # P(T=1 | W=0/1)
P_M_given_T = [0.3, 0.7]   # P(M=1 | T=0/1)
P_Y_given_MU = np.array([[0.05, 0.40],   # P(Y=1 | M=0, U=0/1)
                         [0.30, 0.70]])  # P(Y=1 | M=1, U=0/1)


def mechanism_boxes() -> list[Factor]:
    f_u = Factor(("U",), [1 - P_U, P_U])
    f_w = Factor(("W", "U"),
                 [[1 - P_W_given_U[0], 1 - P_W_given_U[1]],
                  [P_W_given_U[0], P_W_given_U[1]]])
    f_t = Factor(("T", "W"),
                 [[1 - P_T_given_W[0], 1 - P_T_given_W[1]],
                  [P_T_given_W[0], P_T_given_W[1]]])
    f_m = Factor(("M", "T"),
                 [[1 - P_M_given_T[0], 1 - P_M_given_T[1]],
                  [P_M_given_T[0], P_M_given_T[1]]])
    f_y = Factor(("Y", "M", "U"),
                 np.stack([1 - P_Y_given_MU, P_Y_given_MU], axis=0))
    return [f_u, f_w, f_t, f_m, f_y]


def joint(boxes: list[Factor] | None = None) -> Factor:
    boxes = boxes if boxes is not None else mechanism_boxes()
    j = boxes[0]
    for b in boxes[1:]:
        j = j * b
    return j


def main() -> None:
    print("=" * 64)
    print("PART 1 — conditioning vs. surgery")
    print("=" * 64)
    j = joint()

    # P(Y=1 | T=1): slice the T wire at 1, renormalize, discard all but Y
    p_cond = j.condition("T", 1)
    for v in ("U", "W", "M"):
        p_cond = p_cond.marginalize(v)
    p_y_given_t = p_cond.value("Y")

    # P(Y=1 | do(T=1)): CUT the input wire of the T-box — remove its
    # mechanism — and clamp T=1 with a delta box (no input wires)
    boxes_surgery = [b for b in mechanism_boxes() if b.vars[0] != "T"]
    jd = joint(boxes_surgery) * Factor(("T",), [0.0, 1.0])
    for v in ("U", "W", "M", "T"):
        jd = jd.marginalize(v)
    p_y_do_t = jd.value("Y")

    print(f"P(Y=1 | T=1)     (conditioning) : {p_y_given_t:.4f}")
    print(f"P(Y=1 | do(T=1)) (wire surgery) : {p_y_do_t:.4f}")
    assert p_y_given_t > p_y_do_t + 0.02  # confounding gap, exactly computed
    print("Same boxes, same wires — one cut. The gap is confounding, and it is")
    print("VISIBLE in the diagram: COPY(U) feeds both W->T and Y.\n")

    print("=" * 64)
    print("PART 2 — back-door adjustment IS the surgery (theorem, not axiom)")
    print("=" * 64)
    # sum_w P(Y=1 | T=1, W=w) * P(W=w), computed with the same engine
    jt = j.condition("T", 1)
    f_w_marg = joint().marginalize("Y").marginalize("M").marginalize("T").marginalize("U")
    acc = 0.0
    for w in (0, 1):
        jw = jt.condition("W", w)
        for v in ("U", "M"):
            jw = jw.marginalize(v)
        acc += jw.value("Y") * float(f_w_marg.table[w])
    print(f"back-door sum_w P(Y|T=1,W=w)P(W=w) : {acc:.4f}")
    print(f"wire surgery P(Y|do(T=1))          : {p_y_do_t:.4f}")
    assert abs(acc - p_y_do_t) < 1e-10
    print("Exact equality. The back-door criterion is not an extra assumption —")
    print("it is what you get when you push the surgery through the diagram")
    print("(Jacobs, Kissinger & Zanasi 2019; Fritz & Klingler 2023).\n")

    print("=" * 64)
    print("PART 3 — the structural maps: copy and discard")
    print("=" * 64)
    print("COPY(U): U appears in TWO boxes (W|U and Y|M,U). A bare joint")
    print("distribution has no notion of such sharing — two worlds with identical")
    print("joints can differ only in what is copied. That is precisely the")
    print("information an SCM adds over a BN, and precisely what do() exploits.")
    print("DISCARD: every marginalization above is a discard map. Conditioning,")
    print("marginalizing, intervening — three uses of two structural moves.")
    print("\nIn Markov-category language: comonoid (copy) + counit (discard) +")
    print("conditionals as morphisms; do-calculus rules become diagram rewrites.")


if __name__ == "__main__":
    main()
    print("\nLEVEL 5 COMPLETE — all checks passed.")
    print("The bridge is complete: counting -> posteriors -> factor graphs ->")
    print("mechanisms -> error budgets -> composition. The ladder, all the way.")
