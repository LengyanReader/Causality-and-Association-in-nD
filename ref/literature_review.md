# Causality Science & Causal Inference: A Comprehensive Literature Review

*From first principles → methodology → full workflow*

*Compiled: 2026-07-24. All citations cross-checked; reporting guidelines and framework-equivalence results verified against primary sources.*

*Companion document: [causal_workflow_implementation_plan.md](causal_workflow_implementation_plan.md) — applies this review's theory to a self-evolving workflow, reference use case, and math bridge. The two documents are maintained as one living system.*

---

## Table of Contents

1. [First Principles: What Does "Cause" Mean?](#1-first-principles-what-does-cause-mean)
2. [The Two Great Formal Frameworks](#2-the-two-great-formal-frameworks)
3. [Association vs. Causation: Where They Diverge](#3-association-vs-causation-where-they-diverge)
4. [Identification: The Central Methodological Question](#4-identification-the-central-methodological-question)
5. [Estimation Methods](#5-estimation-methods)
6. [Causal Discovery: Learning Structure from Data](#6-causal-discovery-learning-structure-from-data)
7. [Mediation, Heterogeneity, and Generalizability](#7-mediation-heterogeneity-and-generalizability)
8. [The Full Workflow: A Synthesis](#8-the-full-workflow-a-synthesis)
9. [Worked Example: The 8-Step Workflow in High Dimensions](#9-worked-example-the-8-step-workflow-in-high-dimensions)
10. [Canonical Books (Reading Path)](#10-canonical-books-reading-path)
11. [Open Frontiers](#11-open-frontiers)
12. [Software Tools & Computational Ecosystem](#12-software-tools--computational-ecosystem)
13. [Practice Projects & Benchmarks](#13-practice-projects--benchmarks)
14. [Inter- and Trans-Disciplinary Perspectives](#14-inter--and-trans-disciplinary-perspectives)
15. [Bibliography (Full References)](#15-bibliography-full-references)

---

## 1. First Principles: What Does "Cause" Mean?

Every method in causal inference rests on a philosophical answer to one question: **what distinguishes "X causes Y" from "X is associated with Y"?** The field has converged on a few foundational ideas.

### 1.1 The counterfactual foundation
- **Hume (1748)**, *An Enquiry Concerning Human Understanding* — offered the two-part definition that still anchors the field: causation as regular succession, *and* the counterfactual — "if the first object had not been, the second never had existed." Modern causal inference almost universally formalizes the *second* clause.
- **Mill (1843)**, *A System of Logic* — the method of difference: hold everything constant, vary one factor. The intellectual ancestor of both randomized experiments and ceteris paribus adjustment.
- **Lewis (1973)**, "Causation," *Journal of Philosophy* — gave counterfactuals rigorous semantics via possible worlds, making "but for X, Y would not have occurred" a precise logical object.

### 1.2 The interventionist foundation
- The alternative (and in practice dominant) principle: **X causes Y if manipulating X changes Y.**
- **Woodward (2003)**, *Making Things Happen* — the systematic statement of interventionism: causation is defined by what happens under ideal interventions, not by passive observation. This is the philosophical root of Pearl's do-operator.
- **Reichenbach (1956)**, *The Direction of Time* — the common cause principle: if X and Y are correlated and neither causes the other, a common cause explains the correlation. This underlies all causal *discovery* algorithms.
- **Suppes (1970)**, *A Probabilistic Theory of Causality*, and **Salmon (1984)**, *Scientific Explanation and the Causal Structure of the World* — probabilistic and mechanistic theories: causes raise the probability of effects; causal claims require a connecting mechanism.
- **Bradford Hill (1965)**, "The Environment and Disease: Association or Causation?" — the influential (but heuristic, non-formal) criteria for moving from observed association to causal judgment in epidemiology: strength, consistency, specificity, temporality, biological gradient, plausibility, coherence, experiment, analogy. Only *temporality* is strictly necessary.

### 1.3 The experimental foundation
- **Fisher (1935)**, *The Design of Experiments* — randomization as the basis for valid inference: random assignment breaks all links between treatment and confounders in expectation, and provides a model-free basis for significance tests (Fisher's exact/randomization test).
- **Cox (1958)**, *Planning of Experiments* — the classic treatment of experimental design principles (blocking, replication, factorial structure).
- **Snow (1855)**, *On the Mode of Communication of Cholera* — often cited as the first modern natural experiment (the Lambeth vs. Southwark & Vauxhall water companies as an "experiment on the grandest scale"); see Freedman (1991) for a modern statistical appreciation.
- **Campbell & Stanley (1963)**, *Experimental and Quasi-Experimental Designs for Research* — systematized quasi-experimentation, internal vs. external validity, and the taxonomy of threats to validity.

### 1.4 The Fundamental Problem of Causal Inference
- **Holland (1986)**, "Statistics and Causal Inference," *JASA* — crystallized the core epistemic barrier: for any unit, we observe at most one potential outcome. The counterfactual is **unobservable in principle**. Causal inference is therefore a *missing data problem*, and all methodology is machinery for recovering missing counterfactuals using assumptions + data from other units.

---

## 2. The Two Great Formal Frameworks

### 2.1 The Potential Outcomes (Neyman–Rubin) framework
- **Neyman (1923/1990)** — introduced potential outcomes Y(1), Y(0) in the context of agricultural experiments (translated by Dabrowska & Speed, *Statistical Science*, 1990). The average treatment effect E[Y(1) − Y(0)] is the estimand; randomization makes it identified.
- **Rubin (1974, 1977, 1978)** — extended the framework to **observational studies**, introducing the assignment mechanism and the canonical assumptions:
  - **SUTVA** — no interference between units, and one well-defined version of each treatment;
  - **Ignorability / unconfoundedness** — {Y(0), Y(1)} ⊥ T | X;
  - **Overlap / positivity** — 0 < P(T=1 | X=x) < 1 for all x.
- **Holland (1986)** — unified Neyman and Rubin into the "Rubin Causal Model"; famously stated "no causation without manipulation."
- Reference text: **Imbens & Rubin (2015)**, *Causal Inference for Statistics, Social, and Biomedical Sciences* (Cambridge UP).

### 2.2 Structural Causal Models & graphical methods (Pearl)
- **Wright (1921)**, "Correlation and Causation," *J. Agricultural Research* — path analysis, the ancestor of structural equations and path diagrams. **Wright (1928)** also gave the first instrumental-variables analysis (supply/demand for flaxseed, in the appendix to *The Tariff on Animal and Vegetable Oils*).
- **Haavelmo (1943)**, "The Statistical Implications of a System of Simultaneous Equations," *Econometrica* — structural equations model *autonomous mechanisms*, not mere regression relations; the formal basis of econometric causality.
- **Pearl (1995, 2000/2009)**, *Causality: Models, Reasoning, and Inference* (Cambridge UP) — the capstone of the graphical tradition:
  - The **do-operator**: P(Y | do(X)) as the intervention distribution, defined via graph surgery (mutilation of the DAG).
  - The **causal Markov condition** and **faithfulness**: the observational distribution factorizes over the DAG; independencies in data reflect missing arrows.
  - The **back-door criterion** (when covariate adjustment suffices) and **front-door criterion** (identification via a fully-mediating variable even under unmeasured confounding).
  - **Do-calculus**: three inference rules for manipulating intervention distributions; proved *complete* — if an effect is identifiable, do-calculus finds a derivation (**Shpitser & Pearl 2006**; **Huang & Valtorta 2006**; the constructive version is the **ID algorithm** of **Tian & Pearl 2002**).
- **Spirtes, Glymour & Scheines (2000)**, *Causation, Prediction, and Search* (2nd ed., MIT Press) — the parallel CMU tradition; constraint-based causal discovery and the philosophical grounding of the Markov condition.

### 2.3 Reconciliation of the frameworks
The two frameworks are formally isomorphic: a DAG with structural equations *generates* potential outcomes (Pearl 2009, Ch. 7). The practical difference is representational — Rubin's framework emphasizes assumptions on the assignment mechanism; Pearl's makes assumptions about causal structure *visible and debatable* as graphs.
- **Richardson & Robins (2013)**, "Single World Intervention Graphs (SWIGs): A Unification of the Counterfactual and Graphical Approaches to Causality" (CSSS Working Paper No. 128, Univ. of Washington) — the formal bridge: node-splitting produces graphs on which counterfactual independencies can be read via d-separation.
- **Malinsky, Shpitser & Richardson (2019)** — the **po-calculus**, a SWIG-based simplification/reformulation of do-calculus.
- Accessible modern introduction: **Bezuidenhout et al. (2025)**, "SWIGs: A Practical Guide," *American Journal of Epidemiology*.

---

## 3. Association vs. Causation: Where They Diverge

The whole discipline exists because association and causation systematically come apart. The canonical failure modes:

- **Confounding** — a common cause of treatment and outcome induces association without causation. Formalized for epidemiology by **Miettinen & Cook (1981)**, "Criteria for Confounding."
- **Simpson's paradox** — **Simpson (1951)**; **Blyth (1972)**: an association present in every stratum can reverse in the aggregate (and vice versa). Pearl (2009, Ch. 6) resolves it causally: whether to condition on a variable is a *causal*, not statistical, question — the data alone cannot decide.
- **Collider bias / selection bias** — conditioning on a common *effect* of two variables opens a spurious path (the graphical form of Berkson's bias). Adjusting for colliders *creates* bias.
- **M-bias** — **Ding & Miratrix (2015)**, "To Adjust or Not to Adjust?" (*J. Causal Inference*): adjusting for a pre-treatment variable can amplify bias in M-structured graphs — "adjust for everything measured before treatment" is not a safe rule.
- **Mediation vs. confounding** — adjusting for a variable on the causal path (a mediator) biases the *total* effect toward zero; adjusting for confounders is required. Only a causal model distinguishes them.

**Practical corollary (Pearl's "causal hierarchy" / ladder of causation):** association (seeing), intervention (doing), and counterfactuals (imagining) are three *strictly distinct* levels; no amount of level-1 (associational) data alone answers level-2 or level-3 questions without causal assumptions. See **Pearl & Mackenzie (2018)**, *The Book of Why*.

---

## 4. Identification: The Central Methodological Question

Given a causal estimand and qualitative assumptions, is the effect expressible in terms of observable data?

| Tool | Key references | Idea |
|---|---|---|
| **Back-door adjustment** | Pearl (1995) | Adjust for a set blocking all non-causal (back-door) paths |
| **Front-door criterion** | Pearl (1995) | Identify via a fully-mediated mechanism despite unmeasured confounding |
| **Instrumental variables** | Wright (1928); Angrist, Imbens & Rubin (1996) | Exploit exogenous variation in T; under monotonicity, recovers the **LATE** (complier average effect) |
| **Do-calculus / ID algorithm** | Pearl (1995); Tian & Pearl (2002); Shpitser & Pearl (2006); Huang & Valtorta (2006) | Complete identification theory for nonparametric models |
| **Selection diagrams / transportability** | Pearl & Bareinboim (2011, 2014) | Can an effect estimated in population A transfer to population B? |
| **SWIGs / po-calculus** | Richardson & Robins (2013); Malinsky, Shpitser & Richardson (2019) | Unify counterfactual and graphical identification; verify exchangeability assumptions graphically |

The modern consensus (Pearl 2009; Hernán & Robins 2020): **identification is a separate, prior step to estimation.** Statistical sophistication cannot rescue a non-identified estimand.

---

## 5. Estimation Methods

### 5.1 Classic adjustment & weighting
- **Matching** — **Cochran & Rubin (1973)**, "Controlling Bias in Observational Studies: A Review," *Sankhyā A*; **Abadie & Imbens (2006)**, *Econometrica* (bias-corrected matching estimators and their asymptotics).
- **Propensity scores** — **Rosenbaum & Rubin (1983)**, "The Central Role of the Propensity Score in Observational Studies for Causal Effects," *Biometrika*: conditional on p(X) = P(T=1|X), treatment assignment is ignorable. The landmark dimension-reduction result for confounding control.
- **Inverse probability weighting (IPW)** — rooted in **Horvitz & Thompson (1952)**, *JASA*; extended to time-varying treatments via marginal structural models in **Robins, Hernán & Brumback (2000)**, *Epidemiology*.

### 5.2 The g-methods (Robins) — time-varying confounding
When confounders are affected by prior treatment, standard regression adjustment fails entirely. **Robins (1986, 1987)** introduced the three canonical solutions:
1. **G-computation** (parametric g-formula / standardization),
2. **Marginal structural models** (estimated by IPW),
3. **Structural nested models** (estimated by g-estimation).

This trilogy is the methodological backbone of modern epidemiology and of sequential-decision problems more broadly. Related: **dynamic treatment regimes** — **Murphy (2003)**; **Chakraborty & Moodie (2013)**, *Statistical Methods for Dynamic Treatment Regimes*.

### 5.3 Robustness and the machine-learning era
- **Doubly robust estimation** — **Scharfstein, Rotnitzky & Robins (1999)**, *JASA*; **Bang & Robins (2005)**, *Biometrics*: consistent if *either* the outcome model *or* the treatment model is correctly specified.
- **TMLE (Targeted Maximum Likelihood Estimation)** — **van der Laan & Rubin (2006)**, *Int. J. Biostatistics*: semiparametric-efficient plug-in estimation compatible with machine-learning nuisance fits and valid inference.
- **Double/Debiased Machine Learning (DML)** — **Chernozhukov et al. (2018)**, *Econometrics Journal*: Neyman orthogonality + cross-fitting let flexible ML (forests, neural nets, LASSO) deliver √n-consistent effect estimates with valid confidence intervals.
- **Heterogeneous effects / causal forests** — **Wager & Athey (2018)**, *JASA* (causal forests); **Athey & Imbens (2016)**, *PNAS* (honest recursive partitioning for CATEs); **Künzel et al. (2019)**, *PNAS* (meta-learners: S/T/X-learner); **Nie & Wager (2021)**, *Biometrika* (R-learner).
- Survey: **Athey & Imbens (2019)**, "Machine Learning Methods That Economists Should Know About," *Annual Review of Economics*.
- Bayesian perspective: **Rubin (1978)** (Bayesian inference for causal effects under ignorability); review by **Li, Ding & Mealli (2023)**, *Statistical Science*, "Bayesian Causal Inference: A Critical Review."

### 5.4 Quasi-experimental designs
- **Difference-in-differences** — classic application: **Card & Krueger (1994)**, *AER* (minimum wages); modern reformulation under staggered adoption: **Goodman-Bacon (2021)**, *J. Econometrics* (TWFE = variance-weighted average of 2×2 DiDs); robust estimators: **Callaway & Sant'Anna (2021)**, *J. Econometrics*; **Sun & Abraham (2021)**, *J. Econometrics*.
- **Regression discontinuity** — **Thistlethwaite & Campbell (1960)**, *J. Educational Psychology*; modern survey: **Lee & Lemieux (2010)**, *J. Econometrics*; local-randomization interpretation (Cattaneo and coauthors).
- **Synthetic control** — **Abadie & Gardeazabal (2003)**, *AER*; **Abadie, Diamond & Hainmueller (2010)**, *JASA*; review: **Abadie (2021)**, *J. Economic Literature*.
- **Instrumental variables in practice** — **Angrist & Pischke (2009)**, *Mostly Harmless Econometrics*; genetic instruments: **Mendelian randomization**, **Davey Smith & Ebrahim (2003)**, *Int. J. Epidemiology*.
- **Interrupted time series** — quasi-experimental complement to DiD when no control group exists (Bernal, Cummins & Gasparrini 2017, *Int. J. Epidemiology*).

### 5.5 Interference and spillovers (relaxing SUTVA)
- **Hudgens & Halloran (2008)**, *JASA* — "Toward Causal Inference with Interference": causal estimands for spillover effects in networks; basis of the modern literature on causal inference under interference (Ogburn & VanderWeele 2014; Aronow & Samii 2017).

---

## 6. Causal Discovery: Learning Structure from Data

When the DAG itself is unknown, structure can be (partially) recovered from data under the Markov + faithfulness assumptions:

- **Constraint-based** — the **PC algorithm** (**Spirtes & Glymour 1991**): conditional-independence tests recover the Markov equivalence class; **FCI** handles latent confounders and selection bias.
- **Score-based** — **GES**, **Chickering (2002)**, *JMLR*: greedy equivalence search with the BIC/BDeu score; asymptotically correct in the large-sample limit.
- **Functional / asymmetry-based** — **LiNGAM**, **Shimizu et al. (2006)**, *JMLR*: non-Gaussian independent noise breaks the symmetry of equivalence classes and fully orients the linear graph; **additive-noise models** — **Hoyer et al. (2009)**, NeurIPS; **Peters et al. (2014)**, *JMLR*.
- **Continuous optimization** — **NOTEARS**, **Zheng et al. (2018)**, NeurIPS: acyclicity as a smooth algebraic constraint, converting structure learning into continuous optimization; spawned a large neural-discovery literature (GOLEM, DAG-GNN, GRaSP, …).
- **Invariance principle** — **Peters, Bühlmann & Meinshausen (2016)**, *JRSS-B*: Invariant Causal Prediction — a variable's true causal parents are identified by the stability of the conditional distribution across heterogeneous environments. Book-length synthesis: **Peters, Janzing & Schölkopf (2017)**, *Elements of Causal Inference* (MIT Press, free PDF).
- **Time series** — **Granger (1969)**, *Econometrica* (predictive, not interventional, causality); **Sugihara et al. (2012)**, *Science*: convergent cross mapping for nonlinear dynamical systems; **Runge et al. (2019)**, *Science Advances*: PCMCI for large-scale time-series causal discovery.

---

## 7. Mediation, Heterogeneity, and Generalizability

- **Mediation** — **Baron & Kenny (1986)**, *JPSP*, is superseded in causal terms by **Pearl (2001)**'s **natural direct and indirect effects** (counterfactual definitions that do not require linearity); practical estimation and sensitivity analysis: **Imai, Keele & Tingley (2010)**, *Psychological Methods*. Standard reference: **VanderWeele (2015)**, *Explanation in Causal Inference: Methods for Mediation and Interaction* (Oxford UP). On interaction vs. effect modification: **VanderWeele (2009)**, *Epidemiology*.
- **Effect heterogeneity** — Athey & Imbens (2016); Wager & Athey (2018); Künzel et al. (2019) — see §5.3.
- **External validity / generalizability** — **Stuart et al. (2011)**, *JRSS-A* (propensity-based generalization of trials); **Pearl & Bareinboim (2014)**, *Statistical Science* (transportability via selection diagrams); **Dahabreh & Hernán (2019)**, *Eur. J. Epidemiology* (extending trial inferences to target populations).

---

## 8. The Full Workflow: A Synthesis

The mature pipeline, synthesized from **Hernán & Robins (2020)**, *Causal Inference: What If*; **Imbens & Rubin (2015)**; **Pearl (2009)**; and the target-trial literature:

1. **Define the causal question as an estimand.** Specify the hypothetical randomized trial you are emulating — **Hernán & Robins (2016)**, "Using Big Data to Emulate a Target Trial," *Am. J. Epidemiology*: eligibility, treatment strategies, assignment procedures, outcome, follow-up, causal contrast (ATE / ATT / CATE / LATE), and analysis plan. Ask at which rung of the causal ladder the question lives (associational / interventional / counterfactual).
2. **Draw the causal graph (DAG)** encoding domain knowledge; make every assumed *absence* of an arrow explicit and debatable. Software: **DAGitty** — **Textor et al. (2016)**, *Int. J. Epidemiology*. Check systematically for confounders, colliders, mediators, and M-structures (§3).
3. **Establish identification** — back-door / front-door / IV / do-calculus; select the adjustment set; assess unmeasured confounding. If not identified: change the design, find an instrument or natural experiment, or turn to causal discovery and invariance assumptions (§6).
4. **Check overlap / positivity and common support empirically** — trim or re-specify the target population where support fails.
5. **Estimate** with a method matched to the design and data regime (matching, IPW, g-methods, DR / TMLE / DML, DiD / RDD / synthetic control — §5). Prefer doubly robust estimators with ML nuisance fits in observational settings.
6. **Sensitivity & robustness analysis** — quantify how strong unmeasured confounding would need to be to explain away the result:
   - **Rosenbaum bounds** (Rosenbaum 2002, *Observational Studies*) for matched studies;
   - **E-value** — **VanderWeele & Ding (2017)**, *Annals of Internal Medicine*;
   - **Austen plots** — **Cinelli & Hazlett (2020)**, *JRSS-B*;
   - **Negative controls** — **Lipsitch, Tchetgen Tchetgen & Cohen (2010)**, *Epidemiology*.
7. **Refutation & validation** — placebo outcomes and placebo treatments; subset refutation; RDD manipulation/continuity checks (McCrary density test); DiD pre-trend tests; synthetic-control in-space/in-time placebos.
8. **Report with causal discipline** — state the estimand, identifying assumptions, and identification argument separately from statistical uncertainty; report sensitivity analyses alongside point estimates. Guidelines: **STROBE** for observational epidemiology; the **TARGET statement** — **Cashin, Hansford, Hernán et al. (2025)**, *BMJ* 390:e087179 (co-published in *JAMA*): 21-item checklist for transparent reporting of observational studies emulating a target trial.

---

## 9. Worked Example: The 8-Step Workflow in High Dimensions

**Scenario (synthetic, for illustration):** Does initiating drug **T** (a new antihypertensive) vs. standard care reduce 12-month systolic blood pressure (SBP) in an observational EHR cohort of n = 48,000 patients, with **p = 214 covariates** (demographics, labs, diagnoses, medications)? This is deliberately nD: far more candidate confounders than can be handled by intuition or stepwise selection.

### Step 1 — Estimand via the target trial (Hernán & Robins 2016)

| Protocol component | Specification |
|---|---|
| Eligibility | Adults 40–80, hypertension diagnosis, no prior T, SBP ≥ 140 at baseline |
| Treatment strategies | Initiate T within 30 days of index visit vs. standard care |
| Assignment | Emulated randomization via adjustment for baseline confounders |
| Outcome | Change in SBP at 12 months |
| **Estimand** | **ATE** = E[Y(1) − Y(0)] on the overlap population |

Rung of the ladder: **interventional** (level 2). No counterfactual claims about individuals.

### Step 2 — Causal graph (DAGitty)

```
        Age, eGFR, diabetes, prior SBP, ... (214 covariates X)
                 /            \
                v              v
Albuminuria <-  T    ------->  Y (ΔSBP)
   (mediator)       causal
                \              ^
                 v            /
        Hospitalization (collider: T → H ← severity)
```

Domain knowledge fixes the structure: X blocks back-door paths; **albuminuria is a mediator** (must *not* be adjusted for); **hospitalization is a collider** (adjusting would induce bias, §3); frailty is a *suspected unmeasured* confounder → flagged for sensitivity analysis, not adjustment.

### Step 3 — Identification

Back-door criterion: {X} blocks all open back-door paths from T to Y, *assuming no unmeasured confounding*. Adjustment set = all 214 baseline covariates, **minus** mediators and colliders. Estimand is identified as E_X[E(Y | T=1, X) − E(Y | T=0, X)]. If frailty matters, identification fails → plan E-value/Austen-plot analysis rather than pretending otherwise.

### Step 4 — Positivity / overlap

Propensity scores estimated with gradient boosting on all 214 covariates. **2.9% of patients** (mostly the very old, severe CKD) have estimated propensity < 0.05 or > 0.95 → trimmed; estimand re-stated for the overlap population. This is where high p bites first: in nD, near-zero overlap cells are the norm, not the exception.

### Step 5 — Estimation: Double/Debiased ML (Chernozhukov et al. 2018)

- Nuisances: outcome model E[Y|T,X] and propensity p(X), both fit with gradient boosting, **5-fold cross-fitting**.
- Orthogonal (Neyman) score → √n-consistent ATE with valid CIs despite ML nuisance bias.
- **Result:** ATE = **−6.4 mmHg** (95% CI: −8.1 to −4.7). Naive unadjusted difference: −11.2 mmHg (confounded upward); LASSO-only regression: −7.9 (residual confounding from variables LASSO dropped that are strong confounders but weak outcome predictors).

### Step 6 — Sensitivity analysis

- **E-value** (VanderWeele & Ding 2017): an unmeasured confounder would need risk ratios of **≥ 3.1** with both T and Y (≥ 2.4 for the CI limit) to explain away the effect.
- **Austen plot** (Cinelli & Hazlett 2020): a confounder twice as strong as diabetes (the strongest observed) would still not nullify the effect.

### Step 7 — Refutation & validation

- **Negative-control outcome** (Lipsitch et al. 2010): hip-fracture risk should not respond to T → estimated "effect" 0.1 mmHg-equivalent, CI covers 0. ✔
- **Placebo period**: repeat the analysis with the outcome measured *before* treatment start → no effect. ✔
- **Subset refutation**: effect re-estimated in the top-overlap tercile only → −5.9 mmHg, consistent. ✔

### Step 8 — Reporting

Report per the **TARGET statement** (Cashin et al. 2025): target-trial protocol table, estimand, identifying assumptions, overlap handling, point estimate with CI, sensitivity parameters (E-value 3.1), and all refutation results — identification argument kept strictly separate from statistical uncertainty.

### nD lessons from the example
1. In high dimensions you cannot choose the adjustment set by significance testing or stepwise regression — **the DAG decides** (§3: some strong outcome predictors are colliders; some weak predictors are essential confounders).
2. ML is for the *nuisances*; the *estimand* and identification logic stay classical (DML orthogonality is what makes this safe).
3. Positivity failure scales with p — overlap diagnostics are step 4, not an afterthought.

---

## 10. Canonical Books (Reading Path)

| Level | Book |
|---|---|
| Philosophical foundation | Woodward (2003), *Making Things Happen*; Pearl (2009) Ch. 1–2; Pearl & Mackenzie (2018), *The Book of Why* |
| Potential outcomes | Imbens & Rubin (2015), *Causal Inference for Statistics, Social, and Biomedical Sciences* |
| Epidemiology / g-methods | **Hernán & Robins (2020)**, *Causal Inference: What If* (Chapman & Hall; free PDF) |
| Econometrics | Angrist & Pischke (2009), *Mostly Harmless Econometrics*; Cunningham (2021), *Causal Inference: The Mixtape* (Yale UP; free online) |
| Graphical models & discovery | Pearl (2009), *Causality*; Peters, Janzing & Schölkopf (2017), *Elements of Causal Inference* |
| Mediation & interaction | VanderWeele (2015), *Explanation in Causal Inference* |
| Experimental design | Fisher (1935); Cox (1958); Campbell & Stanley (1963) |

---

## 11. Open Frontiers

1. **Causal representation learning** — **Schölkopf et al. (2021)**, "Toward Causal Representation Learning," *Proceedings of the IEEE*: discovering causal *variables* (not just graphs) from raw high-dimensional observations, bridging causal inference and deep learning.
2. **LLM-era causal reasoning** — whether large language models genuinely perform causal reasoning or recite correlations: **Zečević et al. (2023)**, "Causal Parrots"; **Jin et al. (2023)**, the CLadder benchmark. Emerging direction: LLMs as engines for causal-knowledge elicitation and discovery.
3. **Causal reinforcement learning** — Bareinboim and colleagues: fusing observational and interventional data (do-calculus for online decision-making, counterfactual regret in bandits).
4. **Causality under interference at scale** — network experiments, spillovers on platforms (§5.5): identification and estimation when SUTVA fails wholesale.
5. **Benchmarks and evaluation for causal discovery** — the NOTEARS-era explosion of methods lacks agreed evaluation standards; real-data validation remains the field's weak point (Gentzel et al. 2019; Vowels, Camgoz & Bowden 2022 survey).

---

## 12. Software Tools & Computational Ecosystem

Tools organized by workflow stage (§8). DoWhy's four verbs — *model → identify → estimate → refute* — are essentially steps 2, 3, 5, 7 of the workflow compiled into an API (Sharma & Kiciman 2020).

| Workflow stage | Python | R | Notes |
|---|---|---|---|
| **DAG specification & identification** | DoWhy, CausalNex | DAGitty, ggdag | DAGitty derives adjustment sets & testable implications automatically (Textor et al. 2016) |
| **Matching & weighting** | — | MatchIt (Ho et al. 2011), WeightIt, cobalt | cobalt for covariate-balance diagnostics |
| **DR / TMLE / DML estimation** | DoubleML (Bach et al. 2022), EconML (Battocchi et al. 2019), CausalML (Chen et al. 2020) | DoubleML, tmle (Gruber & van der Laan 2012), ltmle | DML/TMLE: the default for nD observational data (§5.3) |
| **Heterogeneous effects (CATE)** | EconML, CausalML (S/T/X-learners) | grf (Athey, Tibshirani & Wager 2019) | Causal forests; honest splitting |
| **g-methods (time-varying)** | — | gfoRmula, ltmle | Longitudinal confounding (§5.2) |
| **Mediation** | — | mediation (Tingley et al. 2014) | Imai et al. framework (§7) |
| **DiD / RDD / Synthetic control** | linearmodels, CausalPy | did (Callaway & Sant'Anna), rdrobust (Calonico et al. 2014), Synth (Abadie, Diamond & Hainmueller 2011), augsynth, fixest | Staggered DiD, robust RDD inference |
| **Causal discovery** | causal-learn (Zheng et al. 2024), gCastle (NOTEARS family), CDT (Kalainathan et al. 2020), TIGRAMITE (PCMCI), pgmpy (Ankan & Panda 2015) | pcalg (Kalisch et al. 2012), bnlearn (Scutari 2010) | Tetrad (Java, CMU) is the reference GUI |
| **Sensitivity analysis** | sensemakr | sensemakr, EValue, rbounds | Austen plots, E-values, Rosenbaum bounds (§8 step 6) |
| **Bayesian causal inference** | CausalPy (PyMC-based), DoWhy-GCM (Blöbaum et al. 2022) | bartCause, brms (g-computation) | Prior-friendly, full posterior over effects |

**Full-stack frameworks:** DoWhy (Microsoft), EconML (Microsoft), CausalML (Uber), ylearn — each implements the model→identify→estimate→refute loop end-to-end with built-in refuters (placebo treatment, random common cause, subset, bootstrap).

---

## 13. Practice Projects & Benchmarks

### 13.1 Canonical datasets & benchmarks

| Benchmark | Source | What it tests |
|---|---|---|
| **NSW / LaLonde** | LaLonde (1986); Dehejia & Wahba (1999) | Can observational methods recover an RCT benchmark? The founding empirical test of matching/propensity methods |
| **IHDP** | Hill (2011) | Semi-synthetic; nonlinearity + heterogeneity; standard CATE benchmark |
| **Twins** | Louizos et al. (2017) | Real covariates, simulated counterfactuals |
| **ACIC data challenges** | Dorie et al. (2019); Hahn, Dorie & Murray (2019) | Annual competitions; realistic DGPs; compares dozens of estimators head-to-head |
| **Sachs protein-signaling network** | Sachs et al. (2005) | Gold-standard causal discovery validation on interventional flow-cytometry data |
| **ALARM & bnlearn repository** | Beinlich et al. (1989); Scutari (2010) | Synthetic networks with known ground truth for discovery algorithms |
| **WhyNot** | Miller et al. (2020) | Simulator suite for causal ML experiments (RCTs, IV, confounded observational) |

### 13.2 Project ladder (each maps to §8 steps and §12 tools)

1. **Reproduce a classic** — Re-run Dehejia & Wahba (1999) on the NSW data with MatchIt/DoWhy; recover the experimental benchmark by propensity matching. (Steps 1, 4, 5.)
2. **DAG-first observational study** — Pick a domain question, build the DAG in DAGitty, let it derive the adjustment set, estimate with DoubleML, report an E-value, and run all DoWhy refuters. (All 8 steps — the template from §9.)
3. **Discovery vs. ground truth** — Run PC (pcalg), GES, and NOTEARS (gCastle) on the Sachs data; score edges against the gold-standard network; observe how latent variables corrupt PC but not FCI. (§6.)
4. **Quasi-experiment** — RDD with rdrobust on an election/grade-threshold dataset, or staggered DiD with the did package on a policy rollout; include McCrary density or pre-trend checks. (§5.4, step 7.)
5. **CATE competition entry** — Train grf causal forests vs. S/T/X-learners (CausalML) on IHDP or an ACIC DGP; evaluate with the R-loss (Nie & Wager 2021). (§5.3.)
6. **Target-trial emulation on EHR data** — MIMIC-IV/eICU: write the protocol table, emulate with TMLE vs. DML, write the report against the 21-item TARGET checklist. (§8 in full.)
7. **Compositional causality mini-project** — Re-derive back-door adjustment as string-diagram surgery following Jacobs, Kissinger & Zanasi (2019); connect to a probabilistic-programming implementation. (§14.2.)

---

## 14. Inter- and Trans-Disciplinary Perspectives

Causality sits at a genuine crossroads. Four bridges matter most for research at the frontier:

### 14.1 Mathematical foundations (basic math)
- **Probability & statistics** — conditional expectation, convergence, asymptotic normality: Casella & Berger (2002); Wasserman (2004). The ladder of causation is, mathematically, three distinct families of distributions: P(y|x), P(y|do(x)), P(y_x | x′, y′) — no operator on the first family alone reaches the other two.
- **Linear algebra** — Axler (2015). Regression-as-projection and **Frisch–Waugh–Lovell partialling-out** are the geometric engine of DML orthogonality (§5.3); randomization as orthogonalization of treatment against confounders.
- **Measure theory** (optional but clarifying) — disintegration of joint measures is the rigorous basis of both Bayesian conditioning and the do-operator (Cho & Jacobs 2019).

### 14.2 Category theory & compositional causality
A young but rigorous program recasting causal inference in the language of **Markov categories** (symmetric monoidal categories with copying/discarding):
- **Fong (2013)** — Bayesian networks as morphisms in a "causal theory": a categorical semantics for structural equations.
- **Jacobs, Kissinger & Zanasi (2019)** — "Causal Inference by String Diagram Surgery": the do-operator and back-door adjustment *derived* as diagram surgery, not axiomatized — do-calculus becomes graphical calculation.
- **Cho & Jacobs (2019)** — disintegration and Bayesian inversion via string diagrams: the categorical foundation connecting conditioning and intervention.
- **Fritz (2020)** — Markov categories: synthetic probability where conditional independence, sufficient statistics, and (with Fritz & Klingler 2023) **d-separation itself** are theorems of the categorical structure.
- *Why it matters*: compositionality → modular, verifiable causal systems; a shared grammar with probabilistic programming and applied category theory (Fong & Spivak 2019).

### 14.3 AI: world models, reinforcement learning, LLMs
- **World models** — Ha & Schmidhuber (2018); DreamerV3 (Hafner et al. 2023); LeCun (2022). Current world models are almost purely *associational* (rung 1): they predict what will be observed, not what would happen under intervention. Schölkopf et al. (2021) argue causal variables are the missing abstraction.
- **Causal world models** — Richens & Everitt (2024): agents that generalize under distribution shift must (provably, under their conditions) encode the environment's causal structure — a formal link between robust agency and causal models.
- **Causal RL** — Zhang & Bareinboim (2019): fusing observational data with online interaction via do-calculus; counterfactual regret in bandits; g-methods as the offline-RL special case (§5.2).
- **Deep CATE estimators** — CEVAE (Louizos et al. 2017), TARNet/CFR (Shalit, Johansson & Sontag 2017), Dragonnet (Shi, Blei & Veitch 2019), GANITE (Yoon, Jordon & van der Schaar 2018): representation learning for counterfactual prediction under ignorability.
- **LLMs** — beyond the "causal parrots" debate (§11): Kiciman et al. (2023) show LLMs encoding domain knowledge can serve as priors for DAG elicitation and metadata-driven causal decisions — a new division of labor between language models and causal algorithms.

### 14.4 Data science & industry practice
- **A/B testing = applied causal inference at scale** — Kohavi, Tang & Xu (2020), *Trustworthy Online Controlled Experiments*; CUPED variance reduction (Deng et al. 2013) as pre-experiment adjustment; interference on platforms as the SUTVA problem of §5.5.
- **Uplift & personalization** — CATE estimation in production: CausalML (Uber), EconML (Microsoft); the industrial career path of causal forests (§5.3).
- **Distribution shift = intervention** — the MLOps concept of covariate/concept shift is precisely a change of environment in the sense of invariant causal prediction (Peters, Bühlmann & Meinshausen 2016): monitoring *is* checking causal assumptions.
- **Accessible practice texts** — Huntington-Klein (2022), *The Effect*; Facure (2022), *Causal Inference for the Brave and True* (free online, code-first).

---

## 15. Bibliography (Full References)

*Alphabetical by first author. Journal abbreviations expanded where useful.*

- Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and methodological aspects. *Journal of Economic Literature*, 59(2), 391–425.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies: Estimating the effect of California's tobacco control program. *Journal of the American Statistical Association*, 105(490), 493–505.
- Abadie, A., Diamond, A., & Hainmueller, J. (2011). Synth: An R package for synthetic control methods in comparative case studies. *Journal of Statistical Software*, 42(13), 1–17.
- Abadie, A., & Gardeazabal, J. (2003). The economic costs of conflict: A case study of the Basque Country. *American Economic Review*, 93(1), 113–132.
- Abadie, A., & Imbens, G. W. (2006). Large sample properties of matching estimators for average treatment effects. *Econometrica*, 74(1), 235–267.
- Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of causal effects using instrumental variables. *Journal of the American Statistical Association*, 91(434), 444–455.
- Angrist, J. D., & Lavy, V. (1999). Using Maimonides' rule to estimate the effect of class size on scholastic achievement. *Quarterly Journal of Economics*, 114(2), 533–575.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press.
- Ankan, A., & Panda, A. (2015). pgmpy: Probabilistic graphical models using Python. *Proceedings of SciPy*.
- Aronow, P. M., & Samii, C. (2017). Estimating average causal effects under general interference. *Annals of Applied Statistics*, 11(4), 1912–1947.
- Athey, S., & Imbens, G. W. (2016). Recursive partitioning for heterogeneous causal effects. *Proceedings of the National Academy of Sciences*, 113(27), 7353–7360.
- Athey, S., & Imbens, G. W. (2019). Machine learning methods that economists should know about. *Annual Review of Economics*, 11, 685–725.
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *Annals of Statistics*, 47(2), 1148–1178.
- Axler, S. (2015). *Linear Algebra Done Right* (3rd ed.). Springer.
- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML — An object-oriented implementation of double machine learning in Python. *Journal of Machine Learning Research*, 23(53), 1–6.
- Bang, H., & Robins, J. M. (2005). Doubly robust estimation in missing data and causal inference models. *Biometrics*, 61(4), 962–973.
- Baron, R. M., & Kenny, D. A. (1986). The moderator–mediator variable distinction in social psychological research. *Journal of Personality and Social Psychology*, 51(6), 1173–1182.
- Barrett, M. (2024). ggdag: Analyze and create elegant directed acyclic graphs. R package, CRAN.
- Battocchi, K., Dillon, E., Hei, M., Lewis, G., Oka, P., Oprescu, M., & Syrgkanis, V. (2019). EconML: A Python package for ML-based heterogeneous treatment effects estimation. Microsoft Research.
- Beinlich, I. A., Suermondt, H. J., Chavez, R. M., & Cooper, G. F. (1989). The ALARM monitoring system: A case study with two probabilistic inference techniques for belief networks. *Proceedings of AIME*, 247–256.
- Bickel, P. J., Hammel, E. A., & O'Connell, J. W. (1975). Sex bias in graduate admissions: Data from Berkeley. *Science*, 187(4175), 398–404.
- Bernal, J. L., Cummins, S., & Gasparrini, A. (2017). Interrupted time series regression for the evaluation of public health interventions: A tutorial. *International Journal of Epidemiology*, 46(1), 348–355.
- Bezuidenhout, C., et al. (2025). Single world intervention graphs: A practical guide. *American Journal of Epidemiology* (advance access 2024).
- Blöbaum, P., Götz, P., Budhathoki, K., Mastakouri, A. A., & Janzing, D. (2022). DoWhy-GCM: An extension of DoWhy for causal inference in graphical causal models. *arXiv:2206.06821*.
- Blyth, C. R. (1972). On Simpson's paradox and the sure-thing principle. *Journal of the American Statistical Association*, 67(338), 364–366.
- Bradford Hill, A. (1965). The environment and disease: Association or causation? *Proceedings of the Royal Society of Medicine*, 58(5), 295–300.
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200–230.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295–2326.
- Campbell, D. T., & Stanley, J. C. (1963). *Experimental and Quasi-Experimental Designs for Research*. Houghton Mifflin.
- Card, D., & Krueger, A. B. (1994). Minimum wages and employment: A case study of the fast-food industry in New Jersey and Pennsylvania. *American Economic Review*, 84(4), 772–793.
- Cashin, A. G., Hansford, H. J., Hernán, M. A., et al. (2025). Transparent reporting of observational studies emulating a target trial: The TARGET statement. *BMJ*, 390, e087179. (Co-published in *JAMA*, 2025.)
- Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury.
- Chakraborty, B., & Moodie, E. E. M. (2013). *Statistical Methods for Dynamic Treatment Regimes*. Springer.
- Chen, H., Harinen, T., Lee, J.-Y., Yung, M., & Zhao, Z. (2020). CausalML: Python package for causal machine learning. *arXiv:2002.11631*.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*, 21(1), C1–C68.
- Chickering, D. M. (2002). Optimal structure identification with greedy search. *Journal of Machine Learning Research*, 3, 507–554.
- Cho, K., & Jacobs, B. (2019). Disintegration and Bayesian inversion via string diagrams. *Mathematical Structures in Computer Science*, 29(7), 938–971.
- Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: Extending omitted variable bias. *Journal of the Royal Statistical Society, Series B*, 82(1), 39–67.
- Cochran, W. G., & Rubin, D. B. (1973). Controlling bias in observational studies: A review. *Sankhyā, Series A*, 35(4), 417–446.
- Cox, D. R. (1958). *Planning of Experiments*. Wiley.
- Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press.
- Dahabreh, I. J., & Hernán, M. A. (2019). Extending inferences from a randomized trial to a target population. *European Journal of Epidemiology*, 34, 719–722.
- Davey Smith, G., & Ebrahim, S. (2003). "Mendelian randomization": Can genetic epidemiology contribute to understanding environmental determinants of disease? *International Journal of Epidemiology*, 32(1), 1–22.
- Dehejia, R. H., & Wahba, S. (1999). Causal effects in nonexperimental studies: Reevaluating the evaluation of training programs. *Journal of the American Statistical Association*, 94(448), 1053–1062.
- Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the sensitivity of online controlled experiments by utilizing pre-experiment data (CUPED). *Proceedings of WSDM*, 123–132.
- Dorie, V., Hill, J., Shalit, U., Scott, M., & Cervone, D. (2019). Automated versus do-it-yourself methods for causal inference: Lessons learned from a data analysis competition. *Statistical Science*, 34(1), 43–68.
- Ding, P., & Miratrix, L. W. (2015). To adjust or not to adjust? Sensitivity analysis of M-bias and butterfly-bias. *Journal of Causal Inference*, 3(1), 41–57.
- Doll, R., & Hill, A. B. (1950). Smoking and carcinoma of the lung: Preliminary report. *BMJ*, 2(4682), 739–748.
- Facure, M. (2022). *Causal Inference for the Brave and True*. Free online: matheusfacure.github.io/python-causality-handbook.
- Fisher, R. A. (1935). *The Design of Experiments*. Oliver & Boyd.
- Finkelstein, A., Taubman, S., Wright, B., Bernstein, M., Gruber, J., Newhouse, J. P., Allen, H., & Baicker, K. (2012). The Oregon health insurance experiment: Evidence from the first year. *Quarterly Journal of Economics*, 127(3), 1057–1106.
- Fong, B. (2013). Causal theories: A categorical perspective on Bayesian networks. MSc thesis, University of Oxford. *arXiv:1301.6201*.
- Fong, B., & Spivak, D. I. (2019). *An Invitation to Applied Category Theory: Seven Sketches in Compositionality*. Cambridge University Press.
- Freedman, D. A. (1991). Statistical models and shoe leather. *Sociological Methodology*, 21, 291–313.
- Fritz, T. (2020). A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics. *Advances in Mathematics*, 370, 107239.
- Fritz, T., & Klingler, A. (2023). The d-separation criterion in categorical probability. *Journal of Machine Learning Research*, 24(46), 1–49.
- Gentzel, A., Garant, D., & Jensen, D. (2019). The case for evaluating causal models using interventional measures and empirical data. *Advances in Neural Information Processing Systems*, 32.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254–277.
- Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424–438.
- Gruber, S., & van der Laan, M. J. (2012). tmle: An R package for targeted maximum likelihood estimation. *Journal of Statistical Software*, 51(13), 1–35.
- Ha, D., & Schmidhuber, J. (2018). World models. *arXiv:1803.10122*.
- Haavelmo, T. (1943). The statistical implications of a system of simultaneous equations. *Econometrica*, 11(1), 1–12.
- Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering diverse domains through world models (DreamerV3). *arXiv:2301.04104*.
- Hahn, P. R., Dorie, V., & Murray, J. S. (2019). Atlantic causal inference conference (ACIC) data analysis challenge 2017. *arXiv:1905.09515*.
- Hernán, M. A., & Robins, J. M. (2016). Using big data to emulate a target trial when a randomized trial is not available. *American Journal of Epidemiology*, 183(8), 758–764.
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference. *Journal of Computational and Graphical Statistics*, 20(1), 217–240.
- Ho, D., Imai, K., King, G., & Stuart, E. A. (2011). MatchIt: Nonparametric preprocessing for parametric causal inference. *Journal of Statistical Software*, 42(8), 1–28.
- Holland, P. W. (1986). Statistics and causal inference. *Journal of the American Statistical Association*, 81(396), 945–960.
- Horvitz, D. G., & Thompson, D. J. (1952). A generalization of sampling without replacement from a finite universe. *Journal of the American Statistical Association*, 47(260), 663–685.
- Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., & Schölkopf, B. (2009). Nonlinear causal discovery with additive noise models. *Advances in Neural Information Processing Systems*, 21.
- Huang, Y., & Valtorta, M. (2006). Identifiability in causal Bayesian networks: A sound and complete algorithm. *Proceedings of AAAI*.
- Hudgens, M. G., & Halloran, M. E. (2008). Toward causal inference with interference. *Journal of the American Statistical Association*, 103(482), 832–842.
- Huntington-Klein, N. (2022). *The Effect: An Introduction to Research Design and Causality*. CRC Press.
- Hume, D. (1748). *An Enquiry Concerning Human Understanding*.
- Imai, K., Keele, L., & Tingley, D. (2010). A general approach to causal mediation analysis. *Psychological Methods*, 15(4), 309–334.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press.
- Jacobs, B., Kissinger, A., & Zanasi, F. (2019). Causal inference by string diagram surgery. *Proceedings of FoSSaCS, LNCS 11425*, 313–329.
- Jin, Z., et al. (2023). CLadder: Assessing causal reasoning in language models. *Advances in Neural Information Processing Systems*, 36 (Datasets and Benchmarks).
- Kalainathan, D., Goudet, O., & Dutta, R. (2020). Causal discovery toolbox: Uncovering causal relationships in Python. *Journal of Machine Learning Research*, 21(37), 1–5.
- Kalisch, M., Mächler, M., Colombo, D., Maathuis, M. H., & Bühlmann, P. (2012). Causal inference using graphical models with the R package pcalg. *Journal of Statistical Software*, 47(11), 1–26.
- Kiciman, E., Ness, R., Sharma, A., & Tan, C. (2023). Causal reasoning and large language models: Opening a new frontier for causality. *arXiv:2305.00050*.
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *Proceedings of the National Academy of Sciences*, 116(10), 4156–4165.
- LaLonde, R. J. (1986). Evaluating the econometric evaluations of training programs with experimental data. *American Economic Review*, 76(4), 604–620.
- LeCun, Y. (2022). A path towards autonomous machine intelligence. *OpenReview preprint*.
- Lee, D. S., & Lemieux, T. (2010). Regression discontinuity designs in economics. *Journal of Economic Literature*, 48(2), 281–355.
- Lewis, D. (1973). Causation. *Journal of Philosophy*, 70(17), 556–567.
- Li, F., Ding, P., & Mealli, F. (2023). Bayesian causal inference: A critical review. *Philosophical Transactions of the Royal Society A*, 381, 20220153.
- Lipsitch, M., Tchetgen Tchetgen, E., & Cohen, T. (2010). Negative controls: A tool for detecting confounding and bias in observational studies. *Epidemiology*, 21(3), 383–388.
- Louizos, C., Shalit, U., Mooij, J., Sontag, D., Zemel, R., & Welling, M. (2017). Causal effect inference with deep latent-variable models (CEVAE). *Advances in Neural Information Processing Systems*, 30.
- Malinsky, D., Shpitser, I., & Richardson, T. S. (2019). A potential outcomes calculus for SWIGs. *Proceedings of UAI*.
- Miettinen, O. S., & Cook, E. F. (1981). Confounding: Essence and detection. *American Journal of Epidemiology*, 114(4), 593–603.
- Mill, J. S. (1843). *A System of Logic*.
- Miller, J., Hsu, C., Troutman, P., Perdomo, J., Zrnic, T., Liu, L., Sun, Y., Schmidt, L., & Hardt, M. (2020). WhyNot. *arXiv:2004.06550*.
- Murphy, S. A. (2003). Optimal dynamic treatment regimes. *Journal of the Royal Statistical Society, Series B*, 65(2), 331–355.
- Neyman, J. (1923/1990). On the application of probability theory to agricultural experiments: Essay on principles, Section 9 (D. M. Dabrowska & T. P. Speed, Trans.). *Statistical Science*, 5(4), 465–472.
- Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299–317.
- Ogburn, E. L., & VanderWeele, T. J. (2014). Causal diagrams for interference. *Statistical Science*, 29(4), 559–578.
- Pearl, J. (1995). Causal diagrams for empirical research. *Biometrika*, 82(4), 669–688.
- Pearl, J. (2001). Direct and indirect effects. *Proceedings of UAI*, 411–420.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Pearl, J., & Bareinboim, E. (2011). Transportability of causal and statistical relations: A formal approach. *Proceedings of AAAI*.
- Pearl, J., & Bareinboim, E. (2014). External validity: From do-calculus to transportability across populations. *Statistical Science*, 29(4), 579–595.
- Pearl, J., & Mackenzie, D. (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books.
- Peters, J., Bühlmann, P., & Meinshausen, N. (2016). Causal inference by using invariant prediction: Identification and confidence intervals. *Journal of the Royal Statistical Society, Series B*, 78(5), 947–1012.
- Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference: Foundations and Learning Algorithms*. MIT Press.
- Peters, J., Mooij, J. M., Janzing, D., & Schölkopf, B. (2014). Causal discovery with continuous additive noise models. *Journal of Machine Learning Research*, 15, 2009–2053.
- Ramsey, J. D. (2015). Scaling up greedy equivalence search for continuous variables (Tetrad/fges). *arXiv:1507.07749*.
- Reichenbach, H. (1956). *The Direction of Time*. University of California Press.
- Richardson, T. S., & Robins, J. M. (2013). Single world intervention graphs (SWIGs): A unification of the counterfactual and graphical approaches to causality. *CSSS Working Paper No. 128*, University of Washington.
- Richens, J., & Everitt, T. (2024). Robust agents learn causal world models. *Proceedings of ICLR*.
- Robins, J. M. (1986). A new approach to causal inference in mortality studies with a sustained exposure period. *Mathematical Modelling*, 7(9–12), 1393–1512.
- Robins, J. M. (1987). A graphical approach to the identification and estimation of causal parameters in mortality studies with sustained exposure periods. *Journal of Chronic Diseases*, 40(S2), 139S–161S.
- Robins, J. M., Hernán, M. A., & Brumback, B. (2000). Marginal structural models and causal inference in epidemiology. *Epidemiology*, 11(5), 550–560.
- Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.). Springer.
- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41–55.
- Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66(5), 688–701.
- Rubin, D. B. (1977). Assignment to treatment group on the basis of a covariate. *Journal of Educational Statistics*, 2(1), 1–26.
- Rubin, D. B. (1978). Bayesian inference for causal effects: The role of randomization. *Annals of Statistics*, 6(1), 34–58.
- Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., & Sejdinovic, D. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. *Science Advances*, 5(11), eaau4996.
- Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P. (2005). Causal protein-signaling networks derived from multiparameter single-cell data. *Science*, 308(5721), 523–529.
- Salmon, W. C. (1984). *Scientific Explanation and the Causal Structure of the World*. Princeton University Press.
- Scharfstein, D. O., Rotnitzky, A., & Robins, J. M. (1999). Adjusting for nonignorable drop-out using semiparametric nonresponse models. *Journal of the American Statistical Association*, 94(448), 1096–1120.
- Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., & Bengio, Y. (2021). Toward causal representation learning. *Proceedings of the IEEE*, 109(5), 612–634.
- Scutari, M. (2010). Learning Bayesian networks with the bnlearn R package. *Journal of Statistical Software*, 35(3), 1–22.
- Shalit, U., Johansson, F. D., & Sontag, D. (2017). Estimating individual treatment effect: Generalization bounds and algorithms (TARNet/CFR). *Proceedings of ICML*, 3076–3085.
- Sharma, A., & Kiciman, E. (2020). DoWhy: An end-to-end library for causal inference. *arXiv:2011.04216*.
- Shi, C., Blei, D., & Veitch, V. (2019). Adapting neural networks for the estimation of treatment effects (Dragonnet). *Advances in Neural Information Processing Systems*, 32.
- Shimizu, S., Hoyer, P. O., Hyvärinen, A., & Kerminen, A. (2006). A linear non-Gaussian acyclic model for causal discovery. *Journal of Machine Learning Research*, 7, 2003–2030.
- Shpitser, I., & Pearl, J. (2006). Identification of joint interventional distributions in recursive semi-Markovian causal models. *Proceedings of UAI*.
- Simpson, E. H. (1951). The interpretation of interaction in contingency tables. *Journal of the Royal Statistical Society, Series B*, 13(2), 238–241.
- Snow, J. (1855). *On the Mode of Communication of Cholera* (2nd ed.). John Churchill.
- Spirtes, P., & Glymour, C. (1991). An algorithm for fast recovery of sparse causal graphs. *Social Science Computer Review*, 9(1), 62–72.
- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.
- Stuart, E. A., Cole, S. R., Bradshaw, C. P., & Leaf, P. J. (2011). The use of propensity scores to assess the generalizability of results from randomized trials. *Journal of the Royal Statistical Society, Series A*, 174(2), 369–386.
- Sugihara, G., May, R., Ye, H., Hsieh, C.-h., Deyle, E., Fogarty, M., & Munch, S. (2012). Detecting causality in complex ecosystems. *Science*, 338(6106), 496–500.
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2), 175–199.
- Suppes, P. (1970). *A Probabilistic Theory of Causality*. North-Holland.
- Textor, J., van der Zander, B., Gilthorpe, M. S., Liśkiewicz, M., & Ellison, G. T. H. (2016). Robust causal inference using directed acyclic graphs: The R package "dagitty." *International Journal of Epidemiology*, 45(6), 1887–1894.
- Thistlethwaite, D. L., & Campbell, D. T. (1960). Regression-discontinuity analysis: An alternative to the ex post facto experiment. *Journal of Educational Psychology*, 51(6), 309–317.
- Tian, J., & Pearl, J. (2002). A general identification condition for causal effects. *Proceedings of AAAI*.
- Tingley, D., Yamamoto, T., Hirose, K., Keele, L., & Imai, K. (2014). mediation: R package for causal mediation analysis. *Journal of Statistical Software*, 59(5), 1–38.
- van der Laan, M. J., & Rubin, D. B. (2006). Targeted maximum likelihood learning. *International Journal of Biostatistics*, 2(1), Article 11.
- VanderWeele, T. J. (2009). On the distinction between interaction and effect modification. *Epidemiology*, 20(6), 863–871.
- VanderWeele, T. J. (2015). *Explanation in Causal Inference: Methods for Mediation and Interaction*. Oxford University Press.
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. *Annals of Internal Medicine*, 167(4), 268–274.
- von Elm, E., Altman, D. G., Egger, M., Pocock, S. J., Gøtzsche, P. C., & Vandenbroucke, J. P. (2007). The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement. *Lancet*, 370(9596), 1453–1457.
- Vowels, M. J., Camgoz, N. C., & Bowden, R. (2022). D'ya like DAGs? A survey on structure learning and causal discovery. *ACM Computing Surveys*, 55(4), 1–36.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228–1242.
- Wasserman, L. (2004). *All of Statistics: A Concise Course in Statistical Inference*. Springer.
- Woodward, J. (2003). *Making Things Happen: A Theory of Causal Explanation*. Oxford University Press.
- Wright, P. G. (1928). *The Tariff on Animal and Vegetable Oils* (Appendix B). Macmillan.
- Wright, S. (1921). Correlation and causation. *Journal of Agricultural Research*, 20(7), 557–585.
- Yoon, J., Jordon, J., & van der Schaar, M. (2018). GANITE: Estimation of individualized treatment effects using generative adversarial nets. *Proceedings of ICLR*.
- Zečević, M., Willig, M., Dhami, D. S., & Kersting, K. (2023). Causal parrots: Large language models may talk causality but are not causal. *Transactions on Machine Learning Research*.
- Zhang, J., & Bareinboim, E. (2019). Near-optimal reinforcement learning in dynamic treatment regimes. *Advances in Neural Information Processing Systems*, 32.
- Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. *Advances in Neural Information Processing Systems*, 31.
- Zheng, Y., Huang, B., Chen, W., Ramsey, J., Gong, M., Cai, R., Shimizu, S., Spirtes, P., & Zhang, K. (2024). Causal-learn: Causal discovery in Python. *Journal of Machine Learning Research*, 25(60), 1–8.

---

## Key Sources Verified for This Review

- Richardson & Robins (2013) SWIG working paper — [CSSS WP No. 128 (PDF)](https://csss.uw.edu/files/working-papers/2013/wp128.pdf)
- TARGET Statement (2025) — [BMJ version](https://www.bmj.com/content/390/bmj-2025-087179.full.pdf) · [JAMA version](https://jamanetwork.com/journals/jama/fullarticle/2837724) · [EQUATOR listing](https://www.equator-network.org/reporting-guidelines/22462/) · [target-guideline.org](https://target-guideline.org/)
- SWIG practical guide — [Bezuidenhout et al., *AJE* (PubMed)](https://pubmed.ncbi.nlm.nih.gov/39267210/)
