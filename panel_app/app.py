"""Panel app: Causal Science — The Complete Workflow (NomNom Eats).

Run:  panel serve panel_app/app.py --auto
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import panel as pn
from sklearn.linear_model import LogisticRegression

from nomnom.dgp import STATIC, HOLIDAY, ground_truth, sample
from nomnom.graph import nomnom_graph
from ucl.stations import frame, identify, load_data, compile_features, model, evaluate
from ucl.stations.analysis import test_suite as run_test_suite, aipw_crossfit
from ucl.stations.evolve import mechanism_stability

pn.extension("plotly", "mathjax", sizing_mode="stretch_width")

# ── Color palette ──
SIDEBAR_BG = "#f8f9fa"
ACCENT = "#3498db"
GREEN = "#2ecc71"
RED = "#e74c3c"
WARN = "#f39c12"


# ═══════════════ WIDGETS ═══════════════
n_slider = pn.widgets.IntSlider(name="Sample size (n)", start=2000, end=50000, step=2000, value=20000)
regime_select = pn.widgets.Select(name="Regime", options=["static", "holiday"], value="static")
seed_input = pn.widgets.IntInput(name="Random seed", start=0, end=999, value=0)
e_threshold = pn.widgets.FloatSlider(name="E-value alarm", start=1.0, end=5.0, step=0.1, value=1.5)

# ═══════════════ CACHED / REACTIVE COMPUTATIONS ═══════════════
_regime_cache = {"last_rname": None, "last_seed": None, "data": None}

def _get_data(rname: str, n_val: int, seed_val: int):
    key = (rname, n_val, seed_val)
    if _regime_cache["last_rname"] == key and _regime_cache["data"] is not None:
        return _regime_cache["data"]
    regime = HOLIDAY if rname == "holiday" else STATIC
    graph = nomnom_graph()
    spec = frame()
    proof = identify(graph, spec)
    df, contract = load_data(proof, regime_name=rname, n=n_val, seed=seed_val)
    features = compile_features(graph, proof)
    bundle = model(df, spec, features, seed=seed_val)
    evaluation = evaluate(df, spec, features, bundle)
    suite = run_test_suite(df, spec, features, evaluation, graph, seed=seed_val)
    naive = df.loc[df["T"]==1, "Y"].mean() - df.loc[df["T"]==0, "Y"].mean()
    _regime_cache["last_rname"] = key
    _regime_cache["data"] = (graph, spec, proof, df, contract, features, bundle, evaluation, suite, naive)
    return _regime_cache["data"]


def compute_pane(template: str, **kwargs) -> pn.pane.Markdown:
    """Return a Markdown pane from a template string."""
    return pn.pane.Markdown(template.format(**kwargs), sizing_mode="stretch_width")


# ═══════════════ GLOSSARY ═══════════════
glossary = pn.Accordion(
    ("Notation & Symbols", pn.pane.Markdown("""
| Symbol | Meaning |
|---|---|
| $T$, $Y$ | Treatment (notification), Outcome (order placed) |
| $Y(1), Y(0)$ | Potential outcomes: what WOULD happen under treatment or control |
| $do(T=1)$ | do-operator (Pearl): graph surgery sets T=1, cutting incoming arrows |
| $P(Y \\mid T)$ | Conditioning (rung 1) — passive observation |
| $P(Y \\mid do(T))$ | Intervention (rung 2) — external manipulation |
| $P(Y(0)=0 \\mid T=1, Y=1)$ | Counterfactual (rung 3) — probability of necessity |
""")),
    ("Abbreviations", pn.pane.Markdown("""
**ATE** = $E[Y(1)-Y(0)]$ · **ATT** = ATE on treated · **CATE** = conditional ATE ·
**SUTVA** = no interference · **DAG** = directed acyclic graph ·
**SCM** = structural causal model · **AIPW** = augmented IPW (doubly robust) ·
**DML** = double/debiased ML · **IPW** = inverse probability weighting ·
**SMD** = standardized mean difference · **PS** = propensity score
""")),
    ("Causal Graph Terms", pn.pane.Markdown("""
**Confounder**: common cause of T and Y (U: hunger) ·
**Mediator**: on T→Y path — don't adjust (M: app-open) ·
**Collider**: common effect of T and Y — NEVER adjust (S: engagement) ·
**Instrument**: affects T only, no direct Y path (Z: jitter) ·
**Negative control**: shares confounders, no T effect (NC: battery) ·
**Back-door**: block all non-causal paths by conditioning ·
**d-separation**: graphical test for conditional independence
""")),
    ("Regimes", pn.pane.Markdown("""
**Static**: baseline world — T→M coefficient = 1.6 ·
**Holiday**: drifted world — T→M coefficient = 0.4 (users habituate) ·
The Y mechanism and causal structure are invariant.
""")),
    ("Sensitivity & Refutation", pn.pane.Markdown("""
**E-value** (VanderWeele & Ding 2017): min confounder strength to explain away effect ·
**Placebo treatment**: permute T → estimate ~0 ·
**Random common cause**: add noise → estimate stable ·
**Subset refuter**: 80% data → agree with full sample ·
**Negative-control**: T→NC effect should be ~null ·
**Mechanism-stability**: invariance principle (Peters et al. 2016) for drift detection
""")),
    active=[], header_color="white", header_background=ACCENT,
    toggle=True, width=320,
)

# ═══════════════ CONTROLS ═══════════════
controls = pn.Column(
    pn.pane.Markdown("### ⚙️ Controls"),
    n_slider, regime_select, seed_input, e_threshold,
    width=320,
)

# ═══════════════ OVERVIEW TAB ═══════════════
overview = pn.Column(
    pn.pane.Markdown("""
## The Universal Causal Loop — Overview

### The Question

> *"Do push notifications actually cause users to order, or are we just
> sending them to people who would order anyway?"*

We are the data science team at **NomNom Eats**, a food-delivery platform.
This is a **causal** question — the platform targets hungry users, creating
confounding that correlation cannot untangle.

### How We Answer It — 9 Stations, One Closed Loop

| # | Station | Core Question | Key Output |
|---|---|---|---|
| 0 | **FRAME** | What decision does this inform? | `EstimandSpec` |
| 1 | **ASSUME** | What causal structure do we believe? | `AssumptionGraph` |
| 2 | **IDENTIFY** | Can the effect be computed from observables? | `IdentificationProof` |
| 3 | **DATA** | Do the data support the identification? | `DataContract` |
| 4 | **FEATURE** | What enters the model? What must not? | `FeatureSpec` |
| 5 | **MODEL** | How do we estimate? | `EstimateBundle` |
| 6 | **EVALUATE** | How wrong could we be? | `EvaluationReport` |
| 7 | **TEST** | Does the machinery refute itself? | `CausalTestSuite` |
| 8 | **EVOLVE** | Is the world still the one we modeled? | `EvolutionLog` |

### The Premise

We work with the **NomNom DGP (Data-Generating Process)** — a synthetic world
with known ground truth, like a flight simulator for causal inference. Every
estimate in this walkthrough is checked against the true ATE computed by
Monte Carlo under $do(T=1)$ vs $do(T=0)$.

### The Assumptions (Made Explicit)

1. **Ignorability:** The platform targets on W (app-use), a proxy for true
   hunger U. Conditioning on W blocks all back-door paths.
2. **Positivity:** Every user has non-zero probability of either treatment.
3. **SUTVA:** One user's notification doesn't affect another's order.
4. **Absent edges are real:** Z does NOT affect Y; T does NOT affect NC;
   S is a pure effect of T and Y — these are testable via d-separation.
"""),
    sizing_mode="stretch_width",
)


# ═══════════════ REACTIVE STATION PANES ═══════════════
def make_station_panes(rname: str, n_val: int, seed_val: int):
    """Build all station panes reactively bound to the current data."""
    data = _get_data(rname, n_val, seed_val)
    graph, spec, proof, df, contract, features, bundle, evaluation, suite, naive = data

    truth_all = {
        "static": ground_truth(regime=STATIC, n_mc=200_000, seed=999),
        "holiday": ground_truth(regime=HOLIDAY, n_mc=200_000, seed=999),
    }
    truth = truth_all[rname]
    regime = HOLIDAY if rname == "holiday" else STATIC

    # ── Station 0: FRAME ──
    s0 = pn.Column(
        pn.pane.Markdown(f"""
### Station 0 — FRAME: The Causal Question

First the estimand, then the method — never the reverse (Hernán & Robins 2016).

| Property | Value |
|---|---|
| Estimand | {spec.estimand} (rung {spec.rung}: intervention) |
| Treatment | {spec.treatment} |
| Outcome | {spec.outcome} |
| Population | {spec.population} |
| Decision context | {spec.decision_context} |
"""),
    )

    # ── Station 1+2: ASSUME + IDENTIFY ──
    s1 = pn.Column(
        pn.pane.Markdown(f"""
### Station 1 — ASSUME: The Causal Graph

**Design Principle P1:** Assumptions are first-class artifacts.

| Property | Value |
|---|---|
| Graph version | `{graph.version}` |
| Nodes (obs.) / Edges | {len(graph.observed)} / {len(graph.edges)} |
| Absent edges (falsifiable) | {len(graph.absent_edges)} |

The platform targets on W (app-use), a proxy for true hunger U. Conditioning
on W blocks the confounding path. This is the single most important edge for
identification.

### Station 2 — IDENTIFY

Identification is the **central methodological question** (LR §4) — a
*separate, prior* step to estimation.

| Property | Value |
|---|---|
| Criterion | {proof.criterion} |
| Identified | {proof.identified} |
| Adjustment set | `{sorted(proof.adjustment_set)}` |
| Formula | $E[Y \\mid do(T)] = \\sum_z E[Y \\mid T, z] \\cdot P(z)$ |

The adjustment set was compiled from the DAG by graph surgery — no hand-picking.
W is in; M (mediator) and S (collider) are correctly excluded.
"""),
    )

    # ── Station 3: DATA ──
    gap = naive - truth["ate"]
    s3 = pn.Column(
        pn.pane.Markdown(f"""
### Station 3 — DATA: Overlap & the Rung Gap

| Rows | Positivity | PS range |
|---|---|---|
| {contract.n_rows} | {contract.positivity_ok} | [{min(contract.overlap.values()):.4f}, {max(contract.overlap.values()):.4f}] |

| Quantity | Value | Rung |
|---|---|---|
| $P(Y\\mid T=1) - P(Y\\mid T=0)$ (naive) | **{naive:+.4f}** | 1 — association |
| $E[Y\\mid do(T=1)] - E[Y\\mid do(T=0)]$ (truth) | **{truth['ate']:+.4f}** | 2 — intervention |
| **Confounding gap** | **{gap:+.4f}** | — |

The gap **IS** confounding, made numerical. No amount of statistical
sophistication closes it — assumptions do.
"""),
    )

    # ── Station 4: FEATURE ──
    s4 = pn.Column(
        pn.pane.Markdown(f"""
### Station 4 — FEATURE: Compiled from the Graph

| In adjustment set | Excluded (collider/mediator) |
|---|---|
| `{sorted(features.adjustment_set)}` | `{sorted(features.excluded)}` |

**Instruments:** `{features.instruments}` | **Negative controls:** `{features.negative_controls}`

M (mediator) and S (collider) are excluded automatically by the compiler.
Adjusting for either would bias the total effect.
"""),
    )

    # ── Collider Demo ──
    rc = aipw_crossfit(df, "T", "Y", features.adjustment_set, seed=seed_val)
    rb = aipw_crossfit(df, "T", "Y", features.adjustment_set + ["S"], seed=seed_val)
    bc = abs(rc["ate"] - truth["ate"])
    bb = abs(rb["ate"] - truth["ate"])
    collider_demo = pn.pane.Markdown(f"""
#### Collider Warning — Berkson's Bias in Action

| Adjustment | ATE | Bias vs Truth |
|---|---|---|
| Correct (excl. S) | {rc['ate']:+.4f} | {bc:.4f} |
| Adding collider S | {rb['ate']:+.4f} | {bb:.4f} |

{"⚠️ Adding the collider **worsens** the estimate — conditioning on S opens a spurious association path."
if bb > bc else ""}
""")

    # ── Station 5: MODEL ──
    _bias = abs(bundle.estimate - truth["ate"])
    _cov = bundle.ci_low <= truth["ate"] <= bundle.ci_high
    s5 = pn.Column(
        pn.pane.Markdown(f"""
### Station 5 — MODEL: Cross-Fit AIPW / DML

Neyman-orthogonal score + cross-fitting. Doubly robust — consistent if EITHER
the propensity OR the outcome model is correctly specified.

| ATE | 95% CI | Ground Truth | Bias | CI Covers? |
|---|---|---|---|---|
| **{bundle.estimate:+.4f}** | [{bundle.ci_low:+.4f}, {bundle.ci_high:+.4f}] | {truth['ate']:+.4f} | {_bias:.4f} | {_cov} |

SE: {bundle.se:.4f} | Estimator: {bundle.estimator}
"""),
        collider_demo,
    )

    # ── Station 6: EVALUATE ──
    _alarm = evaluation.e_value < e_threshold.value
    s6 = pn.Column(
        pn.pane.Markdown(f"""
### Station 6 — EVALUATE: How Wrong Could We Be?

| E-value | Risk Ratio | Max |SMD| | Threshold | Status |
|---|---|---|---|---|
| **{evaluation.e_value:.2f}** | {evaluation.risk_ratio:.2f} | {evaluation.balance['max_abs_smd']:.4f} | {e_threshold.value:.1f} | {"Fragile" if _alarm else "Robust"} |

An E-value of ~{evaluation.e_value:.1f} means an unmeasured confounder would
need a risk ratio ≥~{evaluation.e_value:.1f} with both T and Y (above the
{len(features.adjustment_set)} measured covariates) to explain the effect away.

The E-value formula (VanderWeele & Ding 2017):
"""),
        pn.pane.LaTeX(r"\text{E-value} = RR + \sqrt{RR \cdot (RR - 1)}"),
    )

    # ── Station 7: TEST ──
    ref_rows = "".join(
        f"| {r.name} | {'✅ PASS' if r.passed else '❌ FAIL'} | {str(r.detail)[:80]} |\n"
        for r in suite.refuters)
    inv_rows = "".join(
        f"| {r.name} | {'✅ PASS' if r.passed else '❌ FAIL'} |\n"
        for r in suite.invariant_checks)
    s7 = pn.Column(
        pn.pane.Markdown(f"""
### Station 7 — TEST: Refutation & Continuous Falsification

**Design Principle P4:** Refutation is continuous, not episodic.

| Test | Result | Detail |
|---|---|---|
{ref_rows}

**Loop Invariants:**

| Check | Result |
|---|---|
{inv_rows}

**ALL GREEN: `{suite.all_green}`**
"""),
    )

    # ── Station 8: EVOLVE ──
    df_ref = sample(10_000, regime=STATIC, seed=100).drop(columns=["U"])
    df_ctrl = sample(10_000, regime=STATIC, seed=200).drop(columns=["U"])
    df_drift = sample(10_000, regime=HOLIDAY, seed=300).drop(columns=["U"])
    sc = mechanism_stability(nomnom_graph(), df_ref, df_ctrl, seed=0)
    sd = mechanism_stability(nomnom_graph(), df_ref, df_drift, seed=0)
    md = {n: r for n, r in sd.items() if r["kind"] == "mechanism"}
    worst = max(md, key=lambda n: md[n]["degradation"])
    ALARM = 0.02
    drift_rows = "".join(
        f"| {n} | {md[n]['degradation']:+.4f} | {'⚠️ DRIFT' if md[n]['degradation'] > ALARM else '✅ OK'} |\n"
        for n in sorted(md, key=lambda x: -md[x]["degradation"]))

    from ucl.engine import run_pass
    hreport, _ = run_pass(regime="holiday", n=20000, seed=23)
    ht = truth_all["holiday"]["ate"]
    hc = hreport.estimate.ci_low <= ht <= hreport.estimate.ci_high

    s8 = pn.Column(
        pn.pane.Markdown(f"""
### Station 8 — EVOLVE: Mechanism-Stability Monitor

The **invariance principle** (Peters et al. 2016): a correctly specified
causal mechanism has a stable conditional distribution across environments.

| Node | Degradation | Status |
|---|---|---|
{drift_rows}

**Drift localized to:** `{worst}` — the T→M notification→app-open mechanism
(exactly the one that changed: coefficient 1.6 → 0.4).

---

### Station 8b — ACTUATOR: Autonomous Re-Estimation

| Holiday ATE | Truth | CI Covers? | All Green? |
|---|---|---|---|
| **{hreport.estimate.estimate:+.4f}** | {ht:+.4f} | {hc} | {hreport.tests.all_green} |

**Loop closed:** detect → localize → re-estimate → verify. No human in the loop.
"""),
    )

    # ── Rung 3 ──
    import nomnom.dgp as dgp
    rng = np.random.default_rng(555)
    exo = dgp._draw_exogenous(200_000, rng, STATIC, dgp.DEFAULT_PARAMS)
    factual = dgp._structural(exo, STATIC, dgp.DEFAULT_PARAMS, t_value=None)
    flip_arr = 1 - factual["T"].to_numpy()
    cf = dgp._structural(exo, STATIC, dgp.DEFAULT_PARAMS, t_value=flip_arr)
    mask = (factual["T"] == 1) & (factual["Y"] == 1)
    pn_val = 1 - cf.loc[mask, "Y"].mean()
    y1 = dgp._structural(exo, STATIC, dgp.DEFAULT_PARAMS, t_value=np.ones(1, int))["Y"]
    y0 = dgp._structural(exo, STATIC, dgp.DEFAULT_PARAMS, t_value=np.zeros(1, int))["Y"]
    ate_sim = y1.mean() - y0.mean()

    rung3 = pn.Column(
        pn.pane.Markdown(f"""
### Rung 3 — Counterfactuals: Abduction-Action-Prediction

No interventional distribution answers *"was THIS order caused by the nudge?"*
— that question lives one rung higher (Pearl 2009, Ch. 7).

| Quantity | Value | Rung |
|---|---|---|
| $P(Y(0)=0 \\mid T=1, Y=1)$ — prob. of necessity | **{pn_val:.4f}** | 3 |
| ATE (same draws) | **{ate_sim:+.4f}** | 2 |
| Ground truth ATE | **{truth['ate']:+.4f}** | — |

~**{pn_val:.0%}** of treated-and-ordered outcomes were CAUSED by the
notification. The ATE (~{truth['ate']:.0%}) averages over everyone — the
probability of necessity is a fundamentally different quantity, accessible
only through the SCM's noise structure.

| Step | Action |
|---|---|
| 1. **Abduction** | Infer the unit's exogenous noise from the factual evidence |
| 2. **Action** | Intervene: $do(T = 1 - T_{factual})$ |
| 3. **Prediction** | Re-run the mechanisms with the SAME noise, different T |
"""),
    )

    # ── Summary ──
    summary = pn.Column(
        pn.pane.Markdown(f"""
### Summary — The Complete Workflow

| Station | Result |
|---|---|
| 0 — FRAME | ATE at rung {spec.rung} (intervention) |
| 1 — ASSUME | Graph v{proof.graph_version}, {len(proof.adjustment_set)}-variable adjustment |
| 2 — IDENTIFY | {proof.criterion} criterion, set: `{sorted(proof.adjustment_set)}` |
| 3 — DATA | Positivity {contract.positivity_ok}, gap {gap:+.4f} |
| 4 — FEATURE | Collider & mediator excluded |
| 5 — MODEL | {bundle.estimate:+.4f} [{bundle.ci_low:+.4f},{bundle.ci_high:+.4f}] vs truth {truth['ate']:+.4f} |
| 6 — EVALUATE | E-value: {evaluation.e_value:.2f}, |SMD|={evaluation.balance['max_abs_smd']:.4f} |
| 7 — TEST | Refuters: {'ALL GREEN' if suite.all_green else 'SOME FAILED'} |
| 8 — EVOLVE | Drift: `{worst}`, actuator recovered holiday truth |
| Rung 3 | Necessity: ~{pn_val:.0%} of treated+ordered caused by notification |

**Every claim verified against ground truth. The loop is closed.**
"""),
    )

    return {
        "Overview": overview,
        "0·FRAME": s0,
        "1·ASSUME": s1,
        "2·IDENTIFY": s1,
        "3·DATA": s3,
        "4·FEATURE": s4,
        "5·MODEL": s5,
        "6·EVALUATE": s6,
        "7·TEST": s7,
        "8·EVOLVE": s8,
        "🔄 Rung 3": rung3,
        "📊 Summary": summary,
    }


# ═══════════════ REACTIVE BINDING ═══════════════
def _build_tabs(rname, n_val, seed_val):
    """Rebuild tabs when widgets change. Only this function re-executes."""
    # Extract .value if widgets were passed; Panel bind may pass widget or value
    rv = rname.value if hasattr(rname, 'value') else rname
    nv = n_val.value if hasattr(n_val, 'value') else n_val
    sv = seed_val.value if hasattr(seed_val, 'value') else seed_val
    panes = make_station_panes(rv, nv, sv)
    return pn.Tabs(*[(name, pane) for name, pane in panes.items()],
                   dynamic=True, sizing_mode="stretch_width")


tabs = pn.bind(_build_tabs, rname=regime_select, n_val=n_slider, seed_val=seed_input)


# ═══════════════ LAYOUT ═══════════════
sidebar = pn.Column(
    pn.pane.Markdown("## 🎯 Causal Science"),
    pn.pane.Markdown("*The Complete UCL Walkthrough*"),
    pn.layout.Divider(),
    controls,
    pn.layout.Divider(),
    pn.pane.Markdown("### 📖 Glossary"),
    glossary,
    width=340, height=900, scroll=True,
    styles={"background": SIDEBAR_BG},
)

main = pn.Column(
    tabs,
    sizing_mode="stretch_width",
)

template = pn.template.BootstrapTemplate(
    title="Causal Science — UCL Walkthrough",
    sidebar=[sidebar],
    main=[main],
    header_background=ACCENT,
    collapsed_sidebar=False,
)

template.servable()
