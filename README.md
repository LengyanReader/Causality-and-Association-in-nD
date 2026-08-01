# Causality & Association in nD

*A living research + practice project on causal science — literature review,
self-evolving workflow, ground-truth reference world, interactive demos,
and real-data reproductions, all unit-tested against known truth.*

---

## 🚀 Demos

| Demo | Description | Link |
|---|---|---|
| **Streamlit App** | Interactive walkthrough with persistent glossary sidebar, tabs, expandable sections, live metrics | [→ Open App](https://causality-association-nd.streamlit.app) · `streamlit run streamlit_app/app.py` |
| **Jupyter Notebook** | 37-cell executed notebook: rich markdown, LaTeX, DAG viz, interactive Plotly overview, glossary accordions | [→ View Notebook](notebooks/nomnom_endtoend/full_causal_workflow.ipynb) |
| **NHEFS Real Data** | Same workflow on 1,566 real Americans: does quitting smoking cause weight gain? | [→ View Notebook](notebooks/tier1_gallery/nhefs_real_world_walkthrough.ipynb) |
| **Landing Page** | Project overview website | [→ Open Page](https://lengyanreader.github.io/Causality-and-Association-in-nD) |

---

## Why

Association is not causation — and the gap between them is not a philosophical
footnote but an engineering problem. Three observations:

1. **Causal knowledge is fragmented.** Theory lives in textbooks, estimators in
   libraries, assumptions in analysts' heads. When assumptions aren't artifacts,
   they can't be versioned, tested, or monitored.
2. **Workflows are open loops.** A study runs once: estimate, publish, walk away.
   Production needs the opposite — a closed loop that detects drift and
   re-estimates autonomously.
3. **Correctness is rarely testable.** On real data the truth is unknown. A
   ground-truth world makes the causal stack *unit-testable*.

---

## What's Inside

### 🔄 Universal Causal Loop (`ucl/`)

9 stations — FRAME → ASSUME → IDENTIFY → DATA → FEATURE → MODEL → EVALUATE →
TEST → EVOLVE — each consuming/producing one typed artifact contract. Every
station has **sensors** (quantitative health signals) and **actuators**
(revision actions). The loop is formalized as a control system — it can close
itself.

| # | Station | Core Question | Key Output |
|---|---|---|---|
| 0 | FRAME | What decision does this inform? | `EstimandSpec` |
| 1 | ASSUME | What causal structure do we believe? | `AssumptionGraph` |
| 2 | IDENTIFY | Can the effect be computed from observables? | `IdentificationProof` |
| 3 | DATA | Do the data support the identification? | `DataContract + overlap` |
| 4 | FEATURE | What enters — and what must not? | `FeatureSpec` |
| 5 | MODEL | How do we estimate? | `EstimateBundle + CI` |
| 6 | EVALUATE | How wrong could we be? | `EvaluationReport` |
| 7 | TEST | Does the machinery refute itself? | `CausalTestSuite` |
| 8 | EVOLVE | Is the world still the one we modeled? | `EvolutionLog` |

### 🎯 NomNom Eats (`nomnom/`)

A food-delivery platform as a ground-truth DGP (Data-Generating Process) — like
a flight simulator for causal inference. Contains: confounder (latent hunger U
via proxy W), mediator (app-open M), collider (engagement S), instrument
(send-time jitter Z), negative control (battery NC), RDD (loyalty coupon),
heterogeneous effects (loyal > new), and a holiday regime that changes exactly
one mechanism.

### 📚 Theory (`ref/`)

Two living documents kept in sync: a [literature review](ref/literature_review.md)
(~140 refs, Hume 1748 → TARGET 2025) and an [implementation plan](ref/causal_workflow_implementation_plan.md)
with progress log. Every claim traces to a primary source, a ground-truth test,
or a reproduction.

### 🧪 Causal CI/CD (`causal_ci/`)

37 tests across 4 layers:

| Layer | What it checks |
|---|---|
| **Unit** | DGP sanity, ground-truth recovery (AIPW vs Monte Carlo truth), refuters |
| **Integration** | Graph-compiled: deleting a back-door edge auto-changes adjustment and turns pinned-bias red |
| **Property** | Loop invariants (graph-version provenance, no-descendant adjustment, sensitivity recorded) |
| **Regression** | Pinned ground truths & pipeline outputs — cross-platform tolerances |

### 📐 Math Bridge (`notebooks/math_bridge/`)

6-level curriculum, each level ending by demonstrating which rung it *cannot* reach:

| Level | Topic | Key Insight |
|---|---|---|
| 0 | Foundations | Simpson's paradox — the data alone can't tell you whether to condition on "city" |
| 1 | Bayesian inference | MH from scratch → Bayes conditions but doesn't intervene; the posterior stays on rung 1 |
| 2 | Bayesian networks | Mini-PC skeleton recovery; latent U pollutes NC edges — why FCI exists |
| 3 | The causal step | P(Y\|T) ≠ P(Y\|do(T)); front-door applied without checking criterion fails silently |
| 4 | Estimation theory | Lasso plug-in bias 0.39 vs DML 0.04 — regularization bias in nD, made numerical |
| 5 | Compositional capstone | do() as wire surgery; back-door ≡ surgery to 1e-10 (Jacobs-Kissinger-Zanasi) |

### 📈 Tier-1 Gallery (`notebooks/tier1_gallery/`)

6 classic reproductions on real data:

| Case | Method | Key Result |
|---|---|---|
| Berkeley graduate admissions | Simpson's paradox | Aggregate gap reverses within departments |
| LaLonde/NSW | Propensity matching | Naive −$15,205 → PS match +$2,697 vs $1,794 RCT |
| Card-Krueger | DiD | +2.75 FTE in NJ after minimum wage increase |
| Oregon Medicaid | IV (lottery) | LATE on ED visits and depression |
| Basque terrorism | Synthetic control | GDP gap opens after 1970 |
| Sachs proteins | PC/FCI discovery | 24 edges recovered vs 16 interventional ground truth |

---

## Quickstart

```bash
conda env create -f environment.yml
conda activate causality-nd

# Interactive demo (browser)
streamlit run streamlit_app/app.py

# Jupyter walkthrough
jupyter notebook notebooks/nomnom_endtoend/full_causal_workflow.ipynb

# Full test suite
python -m pytest causal_ci -q   # 37 tests

# Event-driven loop demo
python loops/run_event_loop_demo.py

# Math bridge (any level)
python notebooks/math_bridge/level3_causal_step.py
```

---

## Design Principles

| # | Principle |
|---|---|
| P1 | **Assumptions are first-class artifacts** — versioned DAG with explicit absent edges |
| P2 | **Graph = single source of truth** — adjustment sets, tests, monitors are *compiled* from it |
| P3 | **Sensors + actuators** at every stage — quantitative health signals drive revision actions |
| P4 | **Refutation is continuous** — falsification checks run in CI and monitoring |
| P5 | **Ground truth where possible** — methods graduate to real data after synthetic acceptance |
| P6 | **Ladder discipline** — every query tagged by Pearl's rung; rung-2/3 requires recorded assumptions |

---

## Layout

```
Causality-and-Association-in-nD/
├── ref/                         Living documents (LR + implementation plan)
├── nomnom/                      Ground-truth DGP, causal graph, regimes, episodes
├── ucl/                         UCL engine: contracts, stations, graph compiler, sensors, actuators
├── causal_ci/                   37-test pyramid: unit / integration / property / regression
├── notebooks/
│   ├── math_bridge/             6-level self-verifying curriculum (Levels 0–5)
│   ├── tier1_gallery/           6 classic reproductions + NHEFS real-world walkthrough
│   └── nomnom_endtoend/         Full-walkthrough Jupyter notebook + end-to-end runners
├── streamlit_app/               Interactive demo with persistent glossary sidebar
├── loops/                       Event-driven UCL runner
├── scripts/                     Notebook builder, .ipynb converter
├── docs/                        GitHub Pages landing page
├── .github/workflows/           CI: test pyramid + math bridge + gallery on push
└── runs/                        Generated artifact chains (JSON reports)
```

---

## Status — All Phases Complete ✅

| Phase | Status | Key Evidence |
|---|---|---|
| P0 — Scaffolding | ✅ | Repo layout, artifact contracts with graph-version hashing |
| P1 — First loop pass | ✅ | AIPW/DML recovers true ATE (+0.2437 vs +0.2436); all refuters green |
| P2 — Causal CI pyramid | ✅ | 37 tests; deleting a back-door edge auto-turns the bias test red |
| P3 — Evolution & drift | ✅ | Holiday drift localized to T→M; actuator recovers holiday truth |
| P4 — Loop integration | ✅ | Event-driven UCL: baseline → monitor → alarm → autonomous re-run |
| P5 — Bridge + Gallery | ✅ | 6-level math bridge, 6-gallery reproductions, all green |
