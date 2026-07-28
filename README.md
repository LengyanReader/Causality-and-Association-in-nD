# Causality-and-Association-in-nD

Discussions on Causality, World Modeling, and Spatial-Motion Analysis

A living research + practice project on causality science: a comprehensive
literature review, a self-evolving causal workflow (**UCL — Universal Causal
Loop**), a ground-truth reference world (**NomNom Eats**) that exercises every
component of the workflow, and a **math bridge** from basic probability to
counterfactual reasoning.

---

## Why this project exists

Association is not causation — and the gap between them is not a philosophical
footnote but an engineering problem. Three observations motivate the
architecture:

1. **Causal knowledge is fragmented.** Theory lives in textbooks (LR —
  `ref/literature_review.md`), estimators live in libraries, and assumptions
  live in analysts' heads. When assumptions aren't artifacts, they can't be
  versioned, tested, or monitored — so causal claims silently rot as the world
  changes.
2. **Workflows are open loops.** A typical observational study runs once:
  estimate, publish, walk away. Production systems need the opposite — a
  *closed loop* that keeps checking its own assumptions and re-estimates when
  they break (holiday drift, mechanism change, confounding creep).
3. **Correctness is rarely testable.** On real data the truth is unknown, so
  methodological bugs hide. A ground-truth reference world makes the entire
  causal stack *unit-testable* — the same way flight simulators make autopilot
  software testable.

The project answers with four pillars: **theory** (living documents),
**workflow** (UCL), **world** (NomNom), and **curriculum** (math bridge) —
held together by a causal CI/CD test pyramid.

---

## What the architecture is

```
                        ┌─────────────────────────────┐
                        │  ref/ — LIVING DOCUMENTS    │
                        │  literature_review.md  (LR) │
                        │  implementation_plan.md     │
                        └─────────────┬───────────────┘
                                      │ theory base, cross-referenced everywhere
                                      ▼
   ┌─────────────┐        ┌──────────────────────────────┐        ┌──────────────┐
   │  nomnom/    │  DGP   │  ucl/ — UNIVERSAL CAUSAL LOOP│  tests │  causal_ci/  │
   │  reference  ├───────►│  9 stations, artifact-typed  ├───────►│  test        │
   │  world      │  data  │  (contracts/)                │        │  pyramid     │
   │  (truth!)   │◄───────┤  graph = single source       │        │  37 tests    │
   └─────────────┘  drift│  of truth (compile once,     │        └──────────────┘
                         │  test everywhere)            │
                         └─────────────┬────────────────┘
                                       │ batch events
                                       ▼
                         ┌──────────────────────────────┐
                         │  loops/ — EVENT-DRIVEN UCL   │
                         │  baseline → monitor → alarm  │
                         │  → autonomous actuator re-run│
                         └──────────────────────────────┘

   notebooks/math_bridge/ — the curriculum: Level 0 (foundations) → Level 1
   (Bayes) → Level 2 (Bayesian networks) → Level 3 (SCMs, do(), counterfactuals)
   → Level 4 (estimation theory) → Level 5 (compositional/category theory)
```

### The four pillars

| Pillar | What | Why |
|---|---|---|
| **Theory** (`ref/`) | Two synced living documents: the literature review (~140 refs) and the implementation plan with a progress log | Every claim in code must trace to a cited primary source, a ground-truth test, or a reproduction |
| **Workflow** (`ucl/`) | 9 stations — FRAME, ASSUME, IDENTIFY, DATA, FEATURE, MODEL, EVALUATE, TEST, EVOLVE — each consuming/producing one typed artifact | Formalizes LR §8's 8-step workflow as a **control system**: every station has sensors (health metrics) and actuators (revision actions), so the loop can close itself |
| **World** (`nomnom/`) | NomNom Eats: a food-delivery platform DGP with *known ground truth* — confounder (latent hunger via proxy), instrument, mediator, collider, RDD, negative controls, heterogeneity, and a holiday regime that changes exactly one mechanism | Every concept in the theory appears in one vivid, fully verifiable example; ground truth is computed by Monte Carlo under `do(T)` so estimates can be checked against reality |
| **Curriculum** (`notebooks/math_bridge/`) | Runnable, self-verifying lessons climbing Pearl's ladder | Each level ends by demonstrating which rung it *cannot* reach — the causal gap is the learning objective |

### How the pieces connect

1. **The graph is the single source of truth.** `nomnom/graph.py` declares the
   DAG (with absent edges and rationale). Everything else is *compiled* from it:
   the back-door adjustment set (`ucl/graph_utils.py` — real graph surgery +
   d-separation via networkx), the feature exclusion list (colliders/mediators
   never adjusted), the test battery, and the monitoring checks. Edit the graph
   → the whole pipeline, including its tests, changes automatically (proven by
   the integration tests).
2. **Artifacts carry provenance.** Every contract object carries a
   `graph_version` hash; loop invariant 1 (estimate ↔ identification ↔ graph
   version) is asserted mechanically in every run.
3. **Estimation is honest about its assumptions.** Cross-fit AIPW/DML
   (Neyman-orthogonal, `ucl/stations/analysis.py`) with E-values, IPW balance,
   and a refutation battery (placebo treatment, random common cause, subset,
   negative control) — refutation is continuous, not episodic.
4. **EVOLVE closes the loop.** The mechanism-stability monitor (invariance
   principle) fits P(node | parents) per node and flags which mechanism
   degraded on new data; testable-implication and negative-control monitors run
   alongside. On drift, the actuator re-runs the pass — demonstrated in the
   holiday episode, where the loop localizes the change to the
   notification→app-open mechanism and recovers the new regime's truth
   autonomously.
5. **The test pyramid makes correctness continuous** (`causal_ci/`): unit (DGP
   sanity, ground-truth recovery, refuters), integration (graph-compiled
   consistency), property (loop invariants across seeds/regimes), regression
   (pinned numerical results).

### Design principles (from the plan)

P1 assumptions as first-class artifacts · P2 graph = source of truth · P3 every
stage has sensors and actuators · P4 refutation is continuous · P5 ground truth
where possible, benchmarks where not · P6 ladder discipline (every query tagged
by rung: association / intervention / counterfactual).

---

## Documents (`ref/`)

- [`ref/literature_review.md`](ref/literature_review.md) — theory base: first
  principles, frameworks, identification, estimation, discovery, workflow,
  tools, practice projects, interdisciplinary bridges, full bibliography.
- [`ref/causal_workflow_implementation_plan.md`](ref/causal_workflow_implementation_plan.md)
  — the implementation plan: UCL stations, self-evolution meta-loop, loop/graph
  engineering integration, NomNom use case, math bridge, roadmap, progress log.

## Quickstart

```powershell
conda env create -f environment.yml   # or: conda create -n causality-nd python=3.12 numpy pandas scipy scikit-learn networkx pytest
conda activate causality-nd

# end-to-end UCL pass on the static regime (writes runs/static_pass_report.json)
python notebooks/nomnom_endtoend/run_static_pass.py

# the self-evolving loop: holiday drift detection + autonomous re-run
python notebooks/nomnom_endtoend/run_holiday_episode.py

# production mode: event-driven UCL
python loops/run_event_loop_demo.py

# the math bridge (each script is self-verifying)
python notebooks/math_bridge/level0_foundations.py
python notebooks/math_bridge/level1_bayesian_inference.py
python notebooks/math_bridge/level2_bayesian_networks.py
python notebooks/math_bridge/level3_causal_step.py
python notebooks/math_bridge/level4_estimation_theory.py
python notebooks/math_bridge/level5_compositional_capstone.py

# the Tier-1 gallery: real-data reproductions (downloads & caches NSW data)
python notebooks/tier1_gallery/gallery_berkeley_simpson.py
python notebooks/tier1_gallery/gallery_lalonde_nsw.py

# the full causal CI pyramid (37 tests)
python -m pytest causal_ci -q
```

## Layout

| Path | Contents |
|---|---|
| `ref/` | Living documents (literature review, implementation plan + progress log) |
| `nomnom/` | Ground-truth DGP simulator, regimes, causal graph |
| `ucl/` | Universal Causal Loop: contracts, stations, engine, graph compiler |
| `causal_ci/` | Test pyramid: unit / integration / property / regression |
| `notebooks/math_bridge/` | Curriculum Levels 0–5 (py + .ipynb, self-verifying) |
| `notebooks/tier1_gallery/` | 6 classic reproductions + **real-world walkthrough** (NHEFS: smoking -> weight, complete UCL on 1,566 real Americans) + data loader |
| `notebooks/nomnom_endtoend/` | End-to-end runners (static pass, holiday episode) + **full walkthrough** (`full_causal_workflow.ipynb`) |
| `loops/` | Event-driven UCL runner + demo (production mode) |
| `nomnom/episodes/` | Component demonstrations (RDD coupon, DiD rollout) |
| `ucl/sensors/` | Compilable health sensors per UCL station |
| `ucl/actuators/` | Revision actions encoding the controller policy priority queue |
| `scripts/` | Utilities (py_to_ipynb converter) |
| `.github/workflows/` | CI: full test pyramid + bridge + gallery on push |
| `runs/` | Generated artifact chains (JSON reports) |

## Status

- **P0 scaffolding** ✅ — repo layout, artifact contracts
- **P1 first loop pass** ✅ — NomNom DGP with known ground truth; UCL stations
  0–7 all green; cross-fit AIPW recovers the true ATE (+0.2437 vs +0.2436);
  refuters pass on the valid pipeline and fire on planted bias
- **P2 causal CI pyramid** ✅ — 37 tests across all four layers; deleting a
  back-door edge auto-changes the compiled adjustment set and turns the
  pinned-bias test red
- **P3 discovery & evolution** ✅ — EVOLVE station detects holiday drift,
  localizes it to the notification→app-open mechanism (M), confirms the
  Y-mechanism invariant; actuator re-run recovers the holiday truth
  (+0.1913 vs +0.1910)
- **P4 loop/graph engineering integration** ✅ — event-driven UCL:
  baseline → monitor → drift-alarm → autonomous actuator re-run
- **P5 math bridge + Tier-1 gallery** ✅ — math bridge complete (Levels 0–5).
  Tier-1 gallery: 6 classic reproductions — Berkeley Simpson, LaLonde/NSW,
  Card-Krueger DiD, Oregon lottery IV, Basque synthetic control, Sachs
  protein-signaling discovery. CI config (`causal-ci.yml`), nomnom/episodes
  (RDD + DiD), standalone ucl/sensors/ and ucl/actuators/, and .ipynb
  conversion for all bridge + gallery scripts. **37/37 tests, 6/6 bridge,
  6/6 gallery, 3/3 runners, 2/2 episodes — all green.**
