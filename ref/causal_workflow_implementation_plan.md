# Causal Science Workflow — Implementation Plan

*A general, self-evolving workflow for causality work in research and practice; a complete demonstration use case; and a math bridge from basic probability/Bayesian statistics to causal inference.*

*Status: v1.0 · 2026-07-25 · Living document — kept in sync with [literature_review.md](literature_review.md) (cross-referenced as **LR §n** throughout).*

---

## Table of Contents

1. [Purpose & Design Principles](#1-purpose--design-principles)
2. [The Universal Causal Loop (UCL)](#2-the-universal-causal-loop-ucl)
3. [Self-Evolution: The Meta-Loop](#3-self-evolution-the-meta-loop)
4. [Integration with Development Frameworks](#4-integration-with-development-frameworks)
5. [Demonstration: Examples & The Reference Use Case](#5-demonstration-examples--the-reference-use-case)
6. [The Math Bridge: Basic Math → Bayesian Statistics → Causality](#6-the-math-bridge-basic-math--bayesian-statistics--causality)
7. [Repository & Module Plan](#7-repository--module-plan)
8. [Phased Roadmap with Acceptance Criteria](#8-phased-roadmap-with-acceptance-criteria)
9. [Risk Register & Correctness Checklist](#9-risk-register--correctness-checklist)
10. [References Added in This Document](#10-references-added-in-this-document)

---

## 1. Purpose & Design Principles

**Goal.** Build one workflow that (a) covers the full logic of causal science from LR §8, (b) works for both research (one-shot studies) and practice (production decision systems), (c) self-evolves as data and environments change, and (d) plugs into modern development frameworks — *loop engineering* (agent/verification/event/hill-climbing loops) and *graph engineering* (graph-as-source-of-truth lifecycles).

**Design principles**

| # | Principle | Consequence |
|---|---|---|
| P1 | **Assumptions are first-class artifacts** | Every causal claim carries a versioned, inspectable assumption object (DAG + SCM). No claim without its assumptions. |
| P2 | **The graph is the single source of truth** | Identification, adjustment sets, test suites, and monitoring checks are *compiled from* the DAG, not hand-maintained. Change the graph → regenerate everything. |
| P3 | **Every stage has sensors and actuators** | Each stage emits quantitative health signals (overlap, balance, testable-implication violations, E-values) and has defined revision actions. This makes the loop *closeable*. |
| P4 | **Refutation is continuous, not episodic** | Placebo tests, negative controls, and falsification checks run in CI and in production monitoring, not once at publication time. |
| P5 | **Ground truth where possible; benchmarks where not** | Development happens against synthetic DGPs with known effects (§5.4); methods graduate to real data only after passing synthetic acceptance tests. |
| P6 | **Ladder discipline** | Every query is labeled by its rung on Pearl's ladder (association / intervention / counterfactual — LR §3); the workflow refuses to answer rung-2/3 questions with rung-1 machinery without recorded assumptions. |

---

## 2. The Universal Causal Loop (UCL)

The 8-step workflow of LR §8, extended with a feedback station and formalized as a **control system**: the *setpoint* is a valid, decision-useful causal estimate; the *plant* is the data-generating world; each station has **sensors** (health metrics), **actuators** (revision actions), and a **contract** (input/output artifact).

| # | Station | Core question | Artifact (contract) | Sensors (failure detectors) | Actuators (revision actions) | LR anchor |
|---|---|---|---|---|---|---|
| 0 | **FRAME** | What decision does this inform? What is the target trial? | `EstimandSpec` — estimand (ATE/ATT/CATE/LATE), rung, population, decision context | Ambiguity check: is the estimand policy-relevant and intervention-interpretable? | Reframe question; change rung; narrow population | LR §1, §8.1 |
| 1 | **ASSUME** | What causal structure do we believe? | `AssumptionGraph` — versioned DAG + SCM annotations + explicit *absent* edges | Expert review coverage; assumption debt score (edges asserted without evidence) | Add/remove edges; elicit expert knowledge; consult literature | LR §2.2 |
| 2 | **IDENTIFY** | Is the estimand computable from observables? | `IdentificationProof` — criterion used, adjustment set, or proof of non-identifiability | ID-algorithm result; empty/minimal adjustment sets; M-bias/collider warnings | Change design; seek instrument/natural experiment; move to discovery | LR §4 |
| 3 | **DATA** | Do the data support the identification? | `DataContract` — schema, provenance, overlap report | Positivity violations; common-support gaps; missingness mechanism | Reweight target population; collect more data; trim | LR §8.4 |
| 4 | **FEATURE** | What enters the model — and what must not? | `FeatureSpec` — adjustment set, instruments, negative controls, excluded colliders/mediators | Collider-inclusion alarms; post-treatment leakage detection | Revise feature sets; engineer proxies for latent confounders | LR §3 |
| 5 | **MODEL** | How do we estimate? | `EstimateBundle` — estimator config, nuisance models, point estimate + CI | Balance diagnostics; cross-fitting stability; model class misspecification signals | Switch estimator (matching→DR→TMLE/DML); tune nuisance learners | LR §5 |
| 6 | **EVALUATE** | How wrong could we be? | `EvaluationReport` — sensitivity analyses, E-value, Austen plots, honest uncertainty | E-value below threshold; sensitivity contours crossing zero | Collect negative controls; strengthen design; downgrade claim strength | LR §8.6 |
| 7 | **TEST** | Does the machinery refute itself? | `CausalTestSuite` — refuters, placebos, testable implications, regression tests | Any refuter firing: placebo treatment effect ≠ 0; CI-test violations | Trigger assumption revision (station 1) or re-identification (station 2) | LR §8.7 |
| 8 | **EVOLVE** | Is the world still the one we modeled? | `EvolutionLog` — drift metrics, invariance checks, assumption changelog | Environment-shift detection (ICP residuals); negative-control alarms in production; performance decay | Re-run discovery; re-estimate; re-elicit assumptions; loop to station 1 | LR §6, §8.8 |

**Loop invariants** (must hold at every iteration, checked automatically):
1. Every estimate in `EstimateBundle` is traceable to an `IdentificationProof` from the *current* `AssumptionGraph` version.
2. No feature in `FeatureSpec` is a descendant of treatment unless it is a declared mediator or IV-path variable.
3. `EvaluationReport` exists and its sensitivity parameters are recorded before any claim is published.
4. The `CausalTestSuite` is green on the current data snapshot before deployment.

### 2.1 Control-system reading (why this loop can "self-evolve")
- **Setpoint:** decision loss of acting on the estimate, or claim-validity.
- **Sensors:** the per-station detectors above — all computable from data + graph.
- **Actuators:** the revision actions — all executable without redesigning the loop.
- **Controller policy:** a priority queue — (i) fix failed invariants, (ii) fix failed refuters, (iii) fix positivity, (iv) reduce sensitivity (E-value), (v) reduce variance. This policy is what makes the workflow *context-adaptive*: in an RCT setting the sensors at stations 1–4 fire rarely and the loop is shallow; in observational nD data they fire constantly and the loop spends its budget on identification and sensitivity.

---

## 3. Self-Evolution: The Meta-Loop

The UCL evolves through four mechanisms, ordered from cheapest to most expensive:

1. **Testable-implication monitoring (assumption regression testing).** The DAG implies conditional independencies (d-separation → CI tests, LR §2.2). Compile them into a continuously-run test battery against fresh data. A sustained violation = a falsified edge absence → open an "assumption issue" on `AssumptionGraph` (like a failing unit test opens a bug).
2. **Invariance-based drift detection.** Under invariant causal prediction (LR §6), a correctly-specified causal model has stable residuals across environments. Track environment-tagged residual distributions; instability localizes *which* mechanism changed — and which subgraph to re-learn.
3. **Negative-control alarms.** Pre-registered negative-control outcomes/exposures (LR §8.6) monitored in production; a significant negative-control effect is a smoke alarm for unmeasured confounding drift.
4. **Active experimentation.** When a sensor localizes uncertainty to a specific edge/parameter and the decision value exceeds experiment cost, the loop prescribes a targeted mini-RCT (Thompson-sampled, LR §11 causal RL) — *learning by intervening*, the only mechanism that can create rung-2 evidence directly.

**Assumption versioning.** `AssumptionGraph` lives in git-like version control: every edge change is a commit with rationale + evidence link; every estimate records the graph commit hash. This gives *causal provenance*: "estimate v14 was valid under graph v7; the graph changed at v8 when sensor 2 fired."

---

## 4. Integration with Development Frameworks

### 4.1 Loop engineering
Loop engineering (LangChain's four-loop framework; see §10) structures agentic systems as nested loops: agent loop → verification loop → event-driven loop → hill-climbing loop. The UCL maps onto it directly:

| Loop-engineering level | UCL realization |
|---|---|
| L1 Agent loop (act–observe–reason until done) | A single UCL pass: an analyst or agent walks stations 0–7 with tool calls (DAGitty, DoWhy, EconML — LR §12) |
| L2 Verification loop (score output against rubric, retry with feedback) | Station 6–7 sensors as the rubric: refuters, E-value thresholds, balance checks; failed → feedback to the responsible station |
| L3 Event-driven loop (external events trigger runs) | Data arrival, drift alarms, and experiment completions trigger UCL re-runs automatically |
| L4 Hill-climbing loop (production traces improve the harness) | The meta-loop (§3): assumption commits, estimator choices, and thresholds improve from accumulated `EvolutionLog` traces |

The five elements of a well-engineered loop (testable termination, useful tools, context management, failure exits, adaptive error handling) map to: loop invariants (§2), the LR §12 tool table, the artifact contracts, actuator escalation to human experts, and the controller policy.

### 4.2 Graph engineering
Graph engineering — the lifecycle discipline of systems whose source of truth is a graph (knowledge-graph development processes; see §10) — contributes:
- **Compile-once, test-everywhere:** from `AssumptionGraph`, *generate*: adjustment sets, identification proofs, the CI-test battery, the feature exclusion list, monitoring queries. Nothing is maintained twice.
- **Ontology/KG lifecycle applied to causal graphs:** elicitation (expert + LLM-assisted, LR §14.3) → versioning → validation (testable implications) → evolution (§3) — the KG spiral model with causal semantics.
- **Provenance layer:** the versioned assumption graph doubles as the knowledge graph of what the organization believes about its domain, queryable independently of any single study.

### 4.3 Causal CI/CD — the test pyramid
| Test layer | Contents | Tooling |
|---|---|---|
| **Unit** | Refuters on synthetic DGPs with known effects (placebo treatment, random common cause, subset refutation); estimator correctness vs. ground truth | DoWhy refuters; NomNom DGP (§5.4) |
| **Integration** | Identification ↔ estimation consistency: every identified estimand has an estimator; every excluded feature is a collider/mediator per the graph | Generated from `AssumptionGraph` |
| **Property** | Invariants 1–4 (§2); positivity coverage ≥ threshold; balance SMD < 0.1 | Custom checks |
| **Regression** | Pinned results on benchmark datasets (NSW/LaLonde, IHDP, ACIC — LR §13.1) | pytest + data snapshots |
| **Monitoring** | Testable implications, invariance residuals, negative controls in production | Scheduled jobs + alerts |

---

## 5. Demonstration: Examples & The Reference Use Case

### 5.1 Tier 1 — Real-life micro-example gallery
Vivid canonical cases, each demonstrating specific components (LR anchors in brackets):

| Example | Components demonstrated |
|---|---|
| **Snow's cholera study (1855)** — two water companies as natural experiment | Natural experiment, confounding control by design, mechanism [LR §1.3] |
| **Smoking → lung cancer debate** (Fisher's "constitutional hypothesis" vs. Doll & Hill) | Unmeasured-confounding debate, sensitivity analysis, Bradford Hill criteria [LR §1.2, §8.6] |
| **Berkeley graduate admissions** — Simpson's paradox | Aggregation reversal, mediation vs. confounding [LR §3] |
| **Card & Krueger minimum wage (1994)** | DiD, parallel-trends assumption, refutation via pre-trends [LR §5.4] |
| **Oregon Medicaid health lottery** | Lottery as instrument, LATE, IV assumptions [LR §4, §5.4] |
| **Class size via enrollment thresholds (Angrist & Lavy)** | RDD, local randomization, manipulation checks [LR §5.4] |
| **Abadie & Gardeazabal: terrorism & Basque GDP** | Synthetic control, placebo studies [LR §5.4] |
| **Sachs protein-signaling network** | Causal discovery validated against interventional ground truth [LR §6, §13.1] |
| **Hormone-replacement therapy: Nurses' Health Study vs. WHI RCT** | The canonical observational-vs-RCT failure; immortal-time & confounding lessons motivating target-trial emulation [LR §8.1] |

### 5.2 Tier 2 — The reference use case: **"NomNom Eats"**
One constructed example rich enough to exercise *every* station and component, with **known ground truth** (P5). Setting: a food-delivery platform; core business question: **do push notifications cause orders, and for whom?**

**Component coverage matrix**

| Causal component | NomNom instantiation | UCL stations |
|---|---|---|
| Confounding (observed) | Day-of-week, weather, payday drive both notification response and orders | 1, 2, 4 |
| Unmeasured confounding | User "hunger propensity" (latent) | 6, 7 |
| Collider | "Engagement score" = f(notifications received, orders) — conditioning on it biases everything | 4 (exclusion list) |
| Mediation | Notification → app open → order (total vs. direct effect) | 2, 5 |
| Instrument | Randomized send-time jitter affects open probability, not hunger directly | 2, 5 |
| RDD | Free-delivery coupon at a loyalty-points threshold | 5 |
| DiD | New notification algorithm rolled out city-by-city | 5, 7 |
| Synthetic control | Campaign launched in one metro; donor pool of similar metros | 5, 7 |
| Interference | Friends' orders visible in-app → SUTVA violation | 1, 2, 5 |
| Heterogeneity | Effect by user segment (new vs. loyal) → CATE, causal forests | 5 |
| Time-varying confounding | Weekly notification policy adapting to past user response (g-methods) | 2, 5 |
| Discovery | Full clickstream sensor set; recover the notification funnel graph | 8 |
| Drift / transportability | Holiday season changes mechanisms; model trained pre-holiday fails | 8 |
| Sensitivity/refutation | Negative controls: "notification → battery drain" (null), placebo orders before send | 6, 7 |

### 5.3 NomNom ground-truth SCM (sketch)
The DGP will be implemented as a simulator (`nomnom/`) whose structural equations encode known effects — e.g., a binary treatment with heterogeneous ground-truth effect τ(segment) ∈ {1.0, 3.5} percentage-point order-probability uplift, a front-door-identifiable mediated path, an instrument with known LATE, and an environment switch (holiday regime) that alters exactly one mechanism. Because ground truth is known, every estimator, refuter, and monitor in the UCL can be **unit-tested against truth** (§4.3). Detailed equations are specified in Phase 1 (§8) and live with the code, not in this document.

### 5.4 Why this satisfies "perfect use case"
- **Vivid & relatable** — everyone understands food delivery and push notifications.
- **Complete** — every concept in LR §§1–8 appears at least once (matrix above is the completeness proof; it will be extended whenever LR grows).
- **Verifiable** — synthetic ground truth + real-data analogues (the Tier-1 gallery).
- **Extensible** — new methods get added as new "episodes" (new cities, new regimes) without breaking existing tests.

---

## 6. The Math Bridge: Basic Math → Bayesian Statistics → Causality

A five-level curriculum, each level with notebook deliverables (`notebooks/math_bridge/`). The through-line: **conditioning (seeing) → intervening (doing) → counterfactuals (imagining)** — the same probability machinery, three different operations on it.

### Level 0 — Foundations (LR §14.1)
Sets, probability axioms, conditional probability, independence, expectation, Bayes' rule, common distributions; linear algebra: projection, Frisch–Waugh–Lovell.
*Notebook 0:* Bayes' rule by simulation; Simpson's paradox reproduced in 20 lines — association reverses, causation explains why.

### Level 1 — Bayesian inference
Prior/posterior, conjugacy (Beta–Binomial, Normal–Normal), posterior predictive; computation: grid → MCMC (Metropolis–Hastings) → variational inference; model checking (posterior predictive checks).
*Notebooks 1a–1c:* conjugate updates on NomNom order rates; MH from scratch; the same model in PyMC. **Bridge insight:** Bayesian inference updates beliefs about *parameters of a fixed observational model* — it never leaves rung 1 by itself.

### Level 2 — Bayesian networks
Factorization over DAGs, d-separation, the Markov condition; inference: variable elimination → junction tree → belief propagation; learning: parameter (MLE/Bayes) and structure (constraint: PC; score: GES).
*Notebooks 2a–2b:* d-separation explorer on the NomNom DAG; run PC/GES on simulated data, compare to truth. **Bridge insight:** a BN is a *joint distribution with a graph*; nothing in it yet says "cause" — the same DAG can represent many joint distributions and any orientation in its Markov equivalence class.

### Level 3 — The causal step: BN → SCM
Structural equations as autonomous mechanisms (Haavelmo, LR §2.2); **truncated factorization** (the do-operator): replacing P(x|parents) by δ(x=x₀); interventional vs. conditional distributions computed side by side; back-door and front-door as *theorems about graph surgery*; counterfactuals via abduction–action–prediction; SWIGs unifying with potential outcomes (LR §2.3).
*Notebooks 3a–3c:* P(Y|X) vs. P(Y|do(X)) on the same fitted BN — they differ exactly by confounding; implement back-door adjustment from scratch and against DoWhy; compute a counterfactual ("would this user have ordered without the notification?").
**Bridge theorem set:** truncated factorization; back-door/front-door criteria; do-calculus rules R1–R3; SWIG d-separation ⇒ ignorability statements.

### Level 4 — Estimation theory
From identified functional → estimator: plug-in (g-computation), weighting (IPW/Horvitz–Thompson), semiparametric efficiency (influence functions → doubly robust), Neyman orthogonality + cross-fitting (DML), targeting (TMLE); heterogeneity (R-loss, causal forests).
*Notebooks 4a–4b:* derive the AIPW estimator from the influence function on a toy problem; DML vs. naive ML plug-in bias demo in nD (p >> n) — the core of this project's "nD" theme.
**Bridge insight:** identification is exact and assumption-driven; estimation is approximate and data-driven — the two error budgets must never be conflated (LR §4).

### Level 5 — Compositional capstone (optional/advanced)
Markov categories, disintegration, string-diagram surgery (LR §14.2): the do-operator and back-door adjustment re-derived as diagram rewriting; implementation tie-in with probabilistic programming.
*Notebook 5:* reproduce the smoking/cancer front-door example as string-diagram surgery.

---

## 7. Repository & Module Plan

```
Causality-and-Association-in-nD/
├── ref/
│   ├── literature_review.md          # LR — living review (this plan's theory base)
│   └── causal_workflow_implementation_plan.md   # this document
├── nomnom/                           # reference use case: ground-truth DGP simulator
│   ├── dgp.py                        #   structural equations (single source of truth)
│   ├── regimes.py                    #   environment switches (holiday drift etc.)
│   └── episodes/                     #   one module per component (IV, RDD, DiD, ...)
├── ucl/                              # Universal Causal Loop implementation
│   ├── contracts/                    #   artifact schemas (EstimandSpec, AssumptionGraph, ...)
│   ├── stations/                     #   frame/assume/identify/data/feature/model/evaluate/test/evolve
│   ├── sensors/                      #   health metrics per station
│   ├── actuators/                    #   revision actions
│   └── compile_graph.py              #   graph → adjustment sets, CI-test battery, exclusion lists
├── causal_ci/                        # test pyramid (§4.3)
│   ├── unit/  integration/  property/  regression/
├── notebooks/
│   ├── math_bridge/                  # Level 0–5 notebooks (§6)
│   ├── tier1_gallery/                # real-life case reproductions (§5.1)
│   └── nomnom_endtoend/              # UCL passes on the reference use case
├── loops/                            # loop-engineering integration (event triggers, agent runners)
└── docs/                             # decisions, changelogs, TARGET-style report templates
```

Dependencies (LR §12): DoWhy (identify/estimate/refute), EconML/DoubleML (nD estimation), causal-learn/pcalg (discovery), networkx + DAGitty semantics (graph layer), PyMC (math bridge Level 1–2), pytest + hypothesis (property tests), great-expectations-style data contracts.

---

## 8. Phased Roadmap with Acceptance Criteria

| Phase | Deliverables | Acceptance criteria |
|---|---|---|
| **P0 — Scaffolding** | Repo layout; artifact schemas (`contracts/`); sync convention between this plan and LR | Schemas validated by example instances; LR cross-links resolve |
| **P1 — NomNom DGP + first loop pass** | `dgp.py` with documented ground truth; notebooks running stations 0–7 on the static regime; math-bridge Levels 0–1 | Every UCL artifact produced for ≥1 estimand; DML estimate recovers ground-truth ATE within CI in 95% of 100 seeds; refuters correctly accept/reject planted biases |
| **P2 — Causal CI pyramid** | `causal_ci/` fully wired to generated tests from `compile_graph.py` | Deleting a back-door edge from the graph *automatically* changes the adjustment set and turns a planted-bias test red; CI green on main |
| **P3 — Discovery & evolution** | `stations/evolve`: testable-implication monitor, invariance drift detector, negative-control alarms; holiday-regime episode | Holiday switch is detected by the invariance monitor and localized to the correct mechanism; assumption revision re-opens the loop and restores nominal coverage |
| **P4 — Loop/graph engineering integration** | `loops/` event-driven runner; graph-compiled monitoring dashboards; TARGET-style auto-generated report | A new data batch triggers a full UCL re-run with zero human steps; the report passes the 21-item TARGET checklist (LR §8.8) |
| **P5 — Math bridge completion + Tier-1 gallery** | Notebooks Levels 2–5; ≥5 Tier-1 reproductions | Each notebook derives its headline result from first principles *and* matches the library output; gallery cases cite LR references |

---

## 9. Risk Register & Correctness Checklist

| Risk | Mitigation |
|---|---|
| Workflow overfitting to NomNom (toy-itis) | P5 Tier-1 gallery on real data; ACIC benchmarks in regression tests (LR §13.1) |
| "Automation theatre" — sensors exist but thresholds are arbitrary | Every threshold justified from literature (SMD < 0.1, E-value > effect size, etc.) and logged |
| Graph lock-in: the DAG becomes unfalsifiable dogma | Invariant: absent edges must be *testable and tested*; assumption debt score surfaced in every report |
| Math bridge drifts into generic probability course | Every Level-n notebook must end by demonstrating what rung it *cannot* reach — the causal gap is the learning objective |
| Scope creep across frameworks | UCL contracts are framework-agnostic; DoWhy/EconML are adapters, replaceable per stage |

**Correctness review protocol:** any claim in this repo must trace to (i) an LR-cited primary source, (ii) a NomNom ground-truth test, or (iii) a Tier-1 reproduction — and the trace is recorded in the artifact metadata.

---

## 10. References Added in This Document

(All LR references apply; these are new to this plan.)

- LangChain (2025). *The Art of Loop Engineering*. [langchain.com/blog/the-art-of-loop-engineering](https://www.langchain.com/blog/the-art-of-loop-engineering) — four-loop framework (agent / verification / event-driven / hill-climbing).
- MindStudio (2025). *What Is Loop Engineering? The New Meta for AI Coding Agents*. [mindstudio.ai/blog/what-is-loop-engineering-ai-coding-agents](https://www.mindstudio.ai/blog/what-is-loop-engineering-ai-coding-agents) — five elements of well-engineered loops; ReAct pattern.
- Taminiau, S., et al. *Defining a Knowledge Graph Development Process Through a Systematic Review*. University of Amsterdam. [pure.uva.nl/ws/files/140296592](https://pure.uva.nl/ws/files/140296592/Defining_a_Knowledge_Graph_Development_Process.pdf) — KG lifecycle models (waterfall/V-model/spiral) used in §4.2.
- Angrist, J. D., & Lavy, V. (1999). Using Maimonides' rule to estimate the effect of class size on scholastic achievement. *Quarterly Journal of Economics*, 114(2), 533–575. — RDD example in §5.1.
- Finkelstein, A., et al. (2012). The Oregon health insurance experiment: Evidence from the first year. *Quarterly Journal of Economics*, 127(3), 1057–1106. — IV example in §5.1.
- Bickel, P. J., Hammel, E. A., & O'Connell, J. W. (1975). Sex bias in graduate admissions: Data from Berkeley. *Science*, 187(4175), 398–404. — Simpson's paradox in §5.1.
- Doll, R., & Hill, A. B. (1950). Smoking and carcinoma of the lung. *BMJ*, 2(4682), 739–748. — §5.1.

*Sync note: when this plan adds durable theory or references, mirror them into `literature_review.md` (and vice versa) — the two documents are maintained as one living system.*
