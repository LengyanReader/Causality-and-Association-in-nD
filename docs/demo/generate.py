"""Generate demo_data.json for the interactive browser demo.

Run:  python scripts/generate_demo_data.py

Output: docs/demo_data.json — all UCL results + all explanatory content.
The demo HTML loads this file and renders everything dynamically.
"""

import json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from nomnom.dgp import STATIC, HOLIDAY, ground_truth
from ucl.engine import run_pass
from ucl.stations.evolve import mechanism_stability
from nomnom.dgp import sample as dgp_sample
from nomnom.graph import nomnom_graph

# ── Run the UCL for both regimes ──
def run(regime_name, regime_obj, n=20000, seed=0):
    report, _ = run_pass(regime=regime_name, n=n, seed=seed)
    truth = ground_truth(regime=regime_obj, n_mc=200_000, seed=999)
    return {
        "ate": round(report.estimate.estimate, 4),
        "ci_low": round(report.estimate.ci_low, 4),
        "ci_high": round(report.estimate.ci_high, 4),
        "se": round(report.estimate.se, 4),
        "truth": round(truth["ate"], 4),
        "mu1": round(truth["mu1"], 4),
        "mu0": round(truth["mu0"], 4),
        "cate_loyal": round(truth["cate_loyal"], 4),
        "cate_new": round(truth["cate_new"], 4),
        "e_value": round(report.evaluation.e_value, 2),
        "risk_ratio": round(report.evaluation.risk_ratio, 2),
        "max_smd": round(report.evaluation.balance["max_abs_smd"], 4),
        "positivity_ok": report.data.positivity_ok,
        "n_rows": report.data.n_rows,
        "all_green": report.tests.all_green,
        "criterion": report.identification.criterion,
        "adjustment_set": sorted(report.identification.adjustment_set),
        "graph_version": report.graph.version,
        "estimator": report.estimate.estimator,
        "refuters": [
            {"name": r.name, "passed": r.passed, "detail": str(r.detail)[:120]}
            for r in report.tests.refuters
        ],
    }

data = {"static": run("static", STATIC), "holiday": run("holiday", HOLIDAY)}

# Add holiday actuator
hr, _ = run_pass(regime="holiday", n=20000, seed=23)
ht = ground_truth(regime=HOLIDAY, n_mc=200_000, seed=999)
data["actuator"] = {
    "ate": round(hr.estimate.estimate, 4),
    "truth": round(ht["ate"], 4),
    "all_green": hr.tests.all_green,
    "covers": bool(hr.estimate.ci_low <= ht["ate"] <= hr.estimate.ci_high),
}

# Add naive (observational) values
data["static"]["naive"] = round(
    dgp_sample(50000, seed=0).pipe(
        lambda d: d.loc[d["T"] == 1, "Y"].mean() - d.loc[d["T"] == 0, "Y"].mean()
    ), 4)
data["holiday"]["naive"] = round(
    dgp_sample(50000, regime=HOLIDAY, seed=0).pipe(
        lambda d: d.loc[d["T"] == 1, "Y"].mean() - d.loc[d["T"] == 0, "Y"].mean()
    ), 4)
for rn in ("static", "holiday"):
    data[rn]["gap"] = round(data[rn]["naive"] - data[rn]["truth"], 4)

# Add evolve/drift data
df_ref = dgp_sample(10_000, regime=STATIC, seed=100).drop(columns=["U"])
df_ctrl = dgp_sample(10_000, regime=STATIC, seed=200).drop(columns=["U"])
df_drift = dgp_sample(10_000, regime=HOLIDAY, seed=300).drop(columns=["U"])
sc = mechanism_stability(nomnom_graph(), df_ref, df_ctrl, seed=0)
sd = mechanism_stability(nomnom_graph(), df_ref, df_drift, seed=0)
data["evolve"] = {
    "control": {n: round(r["degradation"], 4) for n, r in sc.items() if r["kind"] == "mechanism"},
    "drift": {n: round(r["degradation"], 4) for n, r in sd.items() if r["kind"] == "mechanism"},
    "worst": max(
        {n: r for n, r in sd.items() if r["kind"] == "mechanism"},
        key=lambda n: sd[n]["degradation"]
    ),
    "alarm": 0.02,
}

# Add rung-3 counterfactual data
import nomnom.dgp as dgp
rng = np.random.default_rng(555)
exo = dgp._draw_exogenous(200_000, rng, STATIC, dgp.DEFAULT_PARAMS)
factual = dgp._structural(exo, STATIC, dgp.DEFAULT_PARAMS, t_value=None)
flip = 1 - factual["T"].to_numpy()
cf = dgp._structural(exo, STATIC, dgp.DEFAULT_PARAMS, t_value=flip)
mask = (factual["T"] == 1) & (factual["Y"] == 1)
y1 = dgp._structural(exo, STATIC, dgp.DEFAULT_PARAMS, t_value=np.ones(1, int))["Y"]
y0 = dgp._structural(exo, STATIC, dgp.DEFAULT_PARAMS, t_value=np.zeros(1, int))["Y"]
data["rung3"] = {
    "p_necessity": round(1 - cf.loc[mask, "Y"].mean(), 4),
    "ate_sim": round(y1.mean() - y0.mean(), 4),
}

# ── Data preview: sample rows from the DGP ──
df_sample = dgp_sample(10, seed=42).drop(columns=["U"])
df_cols = ["T", "Y", "M", "S", "NC", "Z", "W", "rain", "weekend", "payday"]
data["data_preview"] = {
    "columns": df_cols,
    "rows": [list(df_sample.loc[i, df_cols].values) for i in range(10)],
    "shape": {"n_rows": 20000, "n_cols": 13},
    "column_descriptions": {
        "T": "Treatment: notification sent (0/1)",
        "Y": "Outcome: order placed (0/1)",
        "M": "Mediator: app opened (0/1)",
        "S": "Collider: engagement score (0/1)",
        "NC": "Neg. control: battery drain (0/1)",
        "Z": "Instrument: send-time jitter (0/1)",
        "W": "Measured proxy: app-use history (cont.)",
        "rain": "Confounder: raining? (0/1)",
        "weekend": "Confounder: weekend? (0/1)",
        "payday": "Confounder: payday? (0/1)",
    },
}

# ── Data flow diagram: how data transforms through the UCL ──
data["data_flow"] = {
    "stations": [
        {
            "id": "raw", "label": "Raw DGP Sample", "shape": "20,000 x 13",
            "desc": "Observational data from<br>NomNom DGP.<br>T is confounded by U via W.",
            "x": 10, "y": 55, "color": "#3498db"
        },
        {
            "id": "adjust", "label": "Adjustment Set", "shape": "20,000 x 4",
            "desc": "Filter to back-door set:<br>{W, rain, weekend, payday}.<br>Exclude M (mediator),<br>S (collider).",
            "x": 210, "y": 55, "color": "#9b59b6"
        },
        {
            "id": "ps", "label": "Propensity Model", "shape": "e(X) vector",
            "desc": "Fit P(T=1 | X) via<br>gradient boosting.<br>Output: propensity<br>score per row.",
            "x": 410, "y": 55, "color": "#f39c12"
        },
        {
            "id": "outcome", "label": "Outcome Model", "shape": "mu1, mu0 vectors",
            "desc": "Fit E[Y | T, X] via<br>gradient boosting.<br>Predict under both<br>T=1 and T=0.",
            "x": 410, "y": 155, "color": "#1abc9c"
        },
        {
            "id": "aipw", "label": "AIPW Score", "shape": "psi vector",
            "desc": "Compute orthogonal<br>score per row.<br>Cross-fit to avoid<br>overfitting bias.",
            "x": 610, "y": 105, "color": "#2ecc71"
        },
        {
            "id": "ate", "label": "ATE Estimate", "shape": "scalar",
            "desc": "ATE = mean(psi).<br>SE = sd(psi)/sqrt(n).<br>95% CI via normal<br>approximation.",
            "x": 790, "y": 105, "color": "#2ecc71"
        },
    ],
    "arrows": [
        {"from": "raw", "to": "adjust"},
        {"from": "adjust", "to": "ps"},
        {"from": "adjust", "to": "outcome"},
        {"from": "ps", "to": "aipw"},
        {"from": "outcome", "to": "aipw"},
        {"from": "aipw", "to": "ate"},
    ],
}

# ── Content sections (explanatory text, formulas, background) ──
data["content"] = {
    "title": "Causal Science — The Complete Workflow",
    "subtitle": "NomNom Eats: Do push notifications cause user orders?",
    "background": {
        "use_case": "You are the data science team at NomNom Eats, a food-delivery platform. The product manager asks: <i>\"Do push notifications actually cause users to order, or are we just sending them to people who would order anyway?\"</i>",
        "problem": "This is a causal question. The platform's targeting algorithm sends more notifications to users it predicts are hungry — and hungry users order more regardless. The naive association overstates the true effect due to confounding.",
        "premise": "We work with the NomNom DGP (Data-Generating Process) — a synthetic world with known ground truth, like a flight simulator for causal inference. Every estimate is checked against the true ATE computed by Monte Carlo under do(T=1) vs do(T=0).",
        "rungs": "Pearl's ladder of causation: (1) Association: P(Y|T), (2) Intervention: P(Y|do(T)), (3) Counterfactuals: P(Y(0)=0 | T=1, Y=1). This walkthrough climbs all three.",
    },
    "problem_formulation": {
        "question": "Do push notifications cause users to place orders?",
        "why_causal": "Here is what happens when we look at the raw data. The naive difference P(Y|T=1) - P(Y|T=0) is approximately <b>+0.343</b> — suggesting notifications increase order probability by ~34 percentage points. But this number is not the causal effect. It mixes together two fundamentally different things. Here is the breakdown:",
        "confounding_breakdown": [
            "<b>The observed naive difference.</b> P(Y|T=1) − P(Y|T=0) ≈ +0.343. If we naively interpret this as causal, we would conclude notifications boost orders by 34 percentage points.",
            "<b>Component A — the true causal effect.</b> The actual effect of the notification on orders, which we will estimate as ≈ +0.241. This is the quantity we want.",
            "<b>Component B — confounding bias.</b> The spurious difference caused by the platform's targeting: ≈ +0.102. This is the gap between the naive and causal estimates.",
            "<b>Why confounding exists — Mechanism 1 (targeting).</b> True hunger U drives app-use W (U → W), and the platform targets notifications based on W (W → T). So treated users are systematically hungrier than untreated users.",
            "<b>Why confounding exists — Mechanism 2 (outcome).</b> Hunger U also drives orders directly (U → Y). So even without any causal effect of notifications, hungrier (treated) users would order more.",
            "<b>Net result.</b> The treated group has a higher baseline order rate <i>even without any causal effect of notifications</i>. This is the confounding gap — and only causal methods (conditioning on W to block the back-door path U→W→T ... U→Y) can close it.",
        ],
        "target_trial": "The idealized experiment we would run if we could: randomly assign notifications to some users and withhold them from others, then measure the difference in order rates. Since we cannot run this experiment (the platform needs to target), we emulate it from observational data using the back-door criterion.",
        "estimand": "ATE = E[Y(1) - Y(0)] — the expected difference in order probability if every user were notified vs. if no user were notified.",
        "assumptions": [
            "Ignorability: {Y(0), Y(1)} independent of T | {W, rain, weekend, payday} — no unmeasured confounding beyond what W captures",
            "Positivity: 0 < P(T=1 | X=x) < 1 for all covariate patterns x",
            "SUTVA: one user's notification does not affect another user's order",
            "Consistency: the observed outcome under T=t equals the potential outcome Y(t)"
        ],
        "ladder_climbed": "Rung 1 (P(Y|T): naive association) → Rung 2 (P(Y|do(T)): ATE identified via back-door) → Rung 3 (P(Y(0)=0 | T=1, Y=1): probability of necessity via abduction-action-prediction)"
    },
    "dag": {
        "description": "The causal DAG (Directed Acyclic Graph) encodes everything we believe — and everything we DON'T believe — about how notifications and orders relate. Each arrow is an assumption; each absent arrow is a falsifiable claim.",
        "nodes": [
            {"id":"Z","label":"Z (jitter)","x":180,"y":30,"color":"#3498db","role":"Instrument"},
            {"id":"rain","label":"rain","x":180,"y":80,"color":"#95a5a6","role":"Confounder"},
            {"id":"weekend","label":"weekend","x":180,"y":130,"color":"#95a5a6","role":"Confounder"},
            {"id":"payday","label":"payday","x":180,"y":180,"color":"#95a5a6","role":"Confounder"},
            {"id":"U","label":"U (hunger)","x":50,"y":105,"color":"#e8e8e8","role":"Latent confounder"},
            {"id":"W","label":"W (app-use)","x":250,"y":105,"color":"#f39c12","role":"Measured proxy"},
            {"id":"T","label":"T (notify)","x":380,"y":105,"color":"#2ecc40","role":"Treatment"},
            {"id":"M","label":"M (open)","x":510,"y":105,"color":"#8e44ad","role":"Mediator"},
            {"id":"Y","label":"Y (order)","x":640,"y":105,"color":"#2ecc40","role":"Outcome"},
            {"id":"S","label":"S (engage)","x":510,"y":195,"color":"#e67e22","role":"Collider"},
            {"id":"NC","label":"NC (battery)","x":50,"y":195,"color":"#e74c3c","role":"Neg. control"}
        ],
        "edges": [
            {"from":"U","to":"W","style":"dashed","color":"#e74c3c","label":"confounding"},
            {"from":"W","to":"T","style":"dashed","color":"#e74c3c","label":"confounding"},
            {"from":"U","to":"Y","style":"dashed","color":"#e74c3c","label":"confounding"},
            {"from":"U","to":"M","style":"dashed","color":"#e74c3c","label":"confounding"},
            {"from":"U","to":"NC","style":"dashed","color":"#e74c3c","label":"confounding"},
            {"from":"Z","to":"T","style":"dotted","color":"#3498db","label":"instrument"},
            {"from":"rain","to":"T","style":"solid","color":"#95a5a6","label":""},
            {"from":"rain","to":"Y","style":"solid","color":"#95a5a6","label":""},
            {"from":"weekend","to":"T","style":"solid","color":"#95a5a6","label":""},
            {"from":"weekend","to":"Y","style":"solid","color":"#95a5a6","label":""},
            {"from":"payday","to":"T","style":"solid","color":"#95a5a6","label":""},
            {"from":"payday","to":"Y","style":"solid","color":"#95a5a6","label":""},
            {"from":"T","to":"Y","style":"solid","color":"#2ecc40","label":"causal target"},
            {"from":"T","to":"M","style":"dashed","color":"#8e44ad","label":"mediator"},
            {"from":"M","to":"Y","style":"dashed","color":"#8e44ad","label":"mediator"},
            {"from":"T","to":"S","style":"dotted","color":"#e67e22","label":"collider"},
            {"from":"Y","to":"S","style":"dotted","color":"#e67e22","label":"collider"}
        ],
        "legend": [
            {"color":"#2ecc40","style":"solid","label":"Causal target (T->Y)"},
            {"color":"#e74c3c","style":"dashed","label":"Confounding (U paths)"},
            {"color":"#3498db","style":"dotted","label":"Instrument (Z)"},
            {"color":"#8e44ad","style":"dashed","label":"Mediator (M)"},
            {"color":"#e67e22","style":"dotted","label":"Collider (S) — NEVER adjust"},
            {"color":"#95a5a6","style":"solid","label":"Other confounders"}
        ]
    },
    "evalue_deep_dive": {
        "definition": "The E-value (VanderWeele & Ding, 2017, Annals of Internal Medicine) quantifies the minimum strength of association that an unmeasured confounder would need to have with BOTH the treatment and the outcome to fully explain away the observed effect, conditional on the measured covariates.",
        "formula": "E-value = RR_obs + sqrt(RR_obs * (RR_obs - 1)), where RR_obs is the observed risk ratio (or its inverse if < 1)",
        "intuition": "Think of it as a 'worst-case' sensitivity analysis: how strong would a hidden confounder need to be to make our result go away? The larger the E-value, the more robust the finding.",
        "thresholds": [
            "E-value = 1.0: Trivial — any weak confounder could explain the effect away",
            "E-value = 1.0-1.5: Fragile — modest unmeasured confounding could explain the effect",
            "E-value = 1.5-2.0: Moderate — somewhat robust to unmeasured confounding",
            "E-value = 2.0-5.0: Robust — strong unmeasured confounding needed to explain away",
            "E-value > 5.0: Highly robust — very strong unmeasured confounding needed"
        ],
        "ci_bound_note": "For a more conservative assessment, compute the E-value using the confidence interval bound closest to the null (rather than the point estimate). If the CI-bound E-value is still above 2, the conclusion is robust even under parameter uncertainty.",
        "example": "With an E-value of ~2.7 (our static regime estimate), an unmeasured confounder would need to be associated with BOTH notification receipt and order placement by a risk ratio of at least 2.7 — above and beyond the 4 measured covariates (W, rain, weekend, payday) — to reduce the true causal effect to zero. This is a moderately robust finding."
    },
    "stations": [
        {
            "id": "frame", "number": 0, "name": "FRAME", "emoji": "🎯",
            "question": "What decision does this inform?",
            "output": "EstimandSpec",
            "explanation": "First the estimand, then the method — never the reverse (Hernán & Robins 2016). We specify the hypothetical randomized trial we are emulating (target trial): eligibility, treatment strategies, outcome, causal contrast, analysis plan. This question lives on rung 2 (intervention) — it needs do(), not just P(Y|T).",
            "formula": "ATE = E[Y(1) - Y(0)]",
        },
        {
            "id": "assume", "number": 1, "name": "ASSUME", "emoji": "📐",
            "question": "What causal structure do we believe?",
            "output": "AssumptionGraph",
            "explanation": "Assumptions are first-class artifacts (Design Principle P1). Every causal claim carries a versioned, inspectable DAG — and every absent edge is a falsifiable statement about the world. The graph encodes confounders (U→T, U→Y), mediators (T→M→Y), colliders (T→S←Y), instruments (Z→T), and negative controls (NC shares U, no T effect).",
        },
        {
            "id": "identify", "number": 2, "name": "IDENTIFY", "emoji": "🔍",
            "question": "Can the effect be computed from observables?",
            "output": "IdentificationProof",
            "explanation": "Identification is the central methodological question — a separate, prior step to estimation. We use the back-door criterion (Pearl 1995): delete all edges out of T (graph surgery), then find an observed set Z that d-separates T from Y. The adjustment set is compiled from the DAG by graph surgery — no hand-picking. W is in (blocks confounding), M and S are correctly excluded (mediator and collider — never adjust).",
            "formula": "E[Y|do(T)] = Σ_z E[Y | T, z] · P(z)  for z in adjustment set",
        },
        {
            "id": "data", "number": 3, "name": "DATA", "emoji": "📊",
            "question": "Do the data support identification?",
            "output": "DataContract + overlap",
            "explanation": "Even with a correctly identified estimand, the data must support it. The key check is positivity (overlap): every unit must have non-zero probability of receiving either treatment. We also compute the rung-1 vs rung-2 gap — the naive associational contrast vs. the causal truth — to make confounding visible numerically.",
            "metrics": {
                "positivity_threshold": "PS in [0.01, 0.99]",
                "balance_threshold": "|SMD| < 0.1 after IPW weighting",
            },
        },
        {
            "id": "feature", "number": 4, "name": "FEATURE", "emoji": "🔧",
            "question": "What enters the model — and what must not?",
            "output": "FeatureSpec",
            "explanation": "The feature specification is compiled from the graph, not hand-picked. Every exclusion is a graph property: colliders open spurious paths (Berkson's bias), mediators block the causal path (over-adjustment). These are not statistical decisions.",
        },
        {
            "id": "model", "number": 5, "name": "MODEL", "emoji": "🧮",
            "question": "How do we estimate?",
            "output": "EstimateBundle + CI",
            "explanation": "We use AIPW (Augmented Inverse Probability Weighting) with 2-fold cross-fitting — the DML recipe (Chernozhukov et al. 2018). Key properties: (1) Neyman orthogonality — the score function is insensitive to first-order nuisance errors, letting flexible ML (gradient boosting) handle nuisances without contaminating the causal estimand. (2) Double robustness — consistent if EITHER the propensity OR the outcome model is correctly specified.",
            "formula": "ψ = (μ₁ - μ₀) + T(Y - μ₁)/e - (1-T)(Y - μ₀)/(1-e)",
        },
        {
            "id": "evaluate", "number": 6, "name": "EVALUATE", "emoji": "⚠️",
            "question": "How wrong could we be?",
            "output": "EvaluationReport",
            "explanation": "An estimate without a sensitivity analysis is an open-loop claim. We compute two diagnostics: (1) The E-value (VanderWeele & Ding 2017) — the minimum strength an unmeasured confounder would need with BOTH T and Y to explain away the effect, conditional on measured covariates. E-value > 2 = moderately robust. (2) Covariate balance after IPW — |SMD| < 0.1 = adequate balance.",
            "formula": "E-value = RR + √(RR · (RR - 1))",
            "metrics": {
                "E-value interpretation": "1 = trivial, > 2 = moderately robust, > 5 = highly robust",
                "SMD threshold": "< 0.1 = adequate covariate balance",
            },
        },
        {
            "id": "test", "number": 7, "name": "TEST", "emoji": "🧪",
            "question": "Does the machinery refute itself?",
            "output": "CausalTestSuite",
            "explanation": "Refutation is continuous, not episodic (Design Principle P4). The refutation battery applies stress tests: (1) Placebo treatment — permute T randomly, estimate should be ~0. (2) Random common cause — add a random covariate, estimate should be stable. (3) Subset refuter — estimate on 80% of data, should agree with full sample. (4) Negative-control test — estimate T→NC effect, should be ~null (NC shares confounders, no T effect).",
        },
        {
            "id": "evolve", "number": 8, "name": "EVOLVE", "emoji": "🔄",
            "question": "Is the world still the one we modeled?",
            "output": "EvolutionLog",
            "explanation": "The mechanism-stability monitor applies the invariance principle (Peters, Bühlmann & Meinshausen 2016): a correctly specified causal mechanism has a stable conditional distribution across environments. For each endogenous node, we fit P(node | parents) on the reference (static) batch and evaluate log-loss on the new (holiday) batch. The node whose conditional degrades most is the locus of drift. The holiday regime changes exactly one mechanism — T→M (coefficient 1.6→0.4) — and the monitor localizes it correctly.",
        },
    ],
    "rung3_section": {
        "title": "Rung 3 — Counterfactuals: Abduction-Action-Prediction",
        "explanation": "No interventional distribution answers \"was THIS order caused by the nudge?\" — that question lives one rung higher (Pearl 2009, Ch. 7). The three-step recipe: (1) Abduction — infer the unit's exogenous noise from factual evidence. (2) Action — intervene: do(T = 1 - T_factual). (3) Prediction — re-run the mechanisms with the SAME noise, different T. Among treated users who ordered: what fraction was the notification actually necessary for? That is P(Y(0)=0 | T=1, Y=1) — a rung-3 quantity, computable only with the SCM and its noise structure.",
    },
    "actuator_section": {
        "title": "Station 8b — ACTUATOR: Autonomous Re-Estimation",
        "explanation": "The drift detection fires the actuator: re-run the full UCL pass on the new regime with zero human intervention. The same graph, the same identification, the same estimation pipeline — applied to a regime where one mechanism changed. No human re-specified anything. The loop closed: detect → localize → re-estimate → verify.",
    },
    "references": [
        {"short": "Pearl (1995)", "full": "Pearl, J. (1995). Causal diagrams for empirical research. <i>Biometrika</i>, 82(4), 669-688.", "url": "https://doi.org/10.1093/biomet/82.4.669"},
        {"short": "Pearl (2009)", "full": "Pearl, J. (2009). <i>Causality: Models, Reasoning, and Inference</i> (2nd ed.). Cambridge University Press.", "url": "https://doi.org/10.1017/CBO9780511803161"},
        {"short": "Hernan & Robins (2020)", "full": "Hernan, M. A. & Robins, J. M. (2020). <i>Causal Inference: What If</i>. Chapman & Hall/CRC.", "url": "https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/"},
        {"short": "Chernozhukov et al. (2018)", "full": "Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. <i>The Econometrics Journal</i>, 21(1), C1-C68.", "url": "https://doi.org/10.1111/ectj.12097"},
        {"short": "VanderWeele & Ding (2017)", "full": "VanderWeele, T. J. & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. <i>Annals of Internal Medicine</i>, 167(4), 268-274.", "url": "https://doi.org/10.7326/M16-2607"},
        {"short": "Peters et al. (2016)", "full": "Peters, J., Buhlmann, P., & Meinshausen, N. (2016). Causal inference by using invariant prediction: identification and confidence intervals. <i>JRSS-B</i>, 78(5), 947-1012.", "url": "https://doi.org/10.1111/rssb.12167"},
        {"short": "Imbens & Rubin (2015)", "full": "Imbens, G. W. & Rubin, D. B. (2015). <i>Causal Inference for Statistics, Social, and Biomedical Sciences</i>. Cambridge University Press.", "url": "https://doi.org/10.1017/CBO9781139025751"},
        {"short": "Hernan & Robins (2016)", "full": "Hernan, M. A. & Robins, J. M. (2016). Using big data to emulate a target trial when a randomized trial is not available. <i>American Journal of Epidemiology</i>, 183(8), 758-764.", "url": "https://doi.org/10.1093/aje/kwv254"},
    ],
    "glossary": [
        {"term": "ATE", "def": "Average Treatment Effect: E[Y(1) - Y(0)]"},
        {"term": "do(T=1)", "def": "The do-operator (Pearl): graph surgery that sets T=1, cutting all incoming arrows"},
        {"term": "Y(1), Y(0)", "def": "Potential outcomes: what WOULD happen under treatment or control"},
        {"term": "SUTVA", "def": "Stable Unit Treatment Value Assumption: no interference between units, one version of treatment"},
        {"term": "AIPW", "def": "Augmented Inverse Probability Weighting: doubly-robust estimator, Neyman-orthogonal score"},
        {"term": "DML", "def": "Double/Debiased Machine Learning (Chernozhukov et al. 2018): cross-fit orthogonal estimation"},
        {"term": "E-value", "def": "Minimum strength an unmeasured confounder needs with both T and Y to explain away the effect (VanderWeele & Ding 2017)"},
        {"term": "SMD", "def": "Standardized Mean Difference: covariate balance metric; values < 0.1 indicate adequate balance"},
        {"term": "Confounder", "def": "Common cause of T and Y that creates spurious association (U: hunger drives both notifications and orders)"},
        {"term": "Mediator", "def": "Variable on the T→Y causal path — adjusting for it blocks the effect (M: app-open)"},
        {"term": "Collider", "def": "Common effect of T and Y — conditioning on it creates Berkson's bias (S: engagement score)"},
        {"term": "Instrument", "def": "Affects T only, no direct Y path, independent of confounders (Z: randomized send-time jitter)"},
        {"term": "Negative control", "def": "Shares confounders with Y but has no T effect — residual confounding smoke alarm (NC: battery drain)"},
        {"term": "Back-door", "def": "Criterion: block all non-causal paths between T and Y by conditioning on a set Z"},
        {"term": "d-separation", "def": "Graphical test for conditional independence; tests which absent edges are falsifiable"},
        {"term": "Ignorability", "def": "Key assumption: {Y(0), Y(1)} independent of T given X — no unmeasured confounding"},
        {"term": "Positivity", "def": "Every unit has non-zero probability of receiving either treatment; checked at Station 3"},
    ],
}

# ── Write ──
out_path = Path(__file__).resolve().parent / "data.json"
out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Written: {out_path} ({out_path.stat().st_size:,} bytes)")
print(f"  static ATE={data['static']['ate']}, holiday ATE={data['holiday']['ate']}")
print(f"  {len(data['content']['stations'])} stations, {len(data['content']['glossary'])} glossary terms")
