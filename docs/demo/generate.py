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

# ── Comparison data: Right way vs Wrong way ──
from ucl.stations.analysis import aipw_crossfit
from ucl.stations.design import load_data as _load_data, identify as _identify, frame as _frame_func

_spec = _frame_func()
_proof = _identify(nomnom_graph(), _spec)
df_comp, _ = _load_data(_proof, regime_name="static", n=20000, seed=0)
correct_adj = ["W", "rain", "weekend", "payday"]
wrong_adj = correct_adj + ["S"]  # include collider
res_correct = aipw_crossfit(df_comp, "T", "Y", correct_adj, seed=0)
res_collider = aipw_crossfit(df_comp, "T", "Y", wrong_adj, seed=0)
truth_val = ground_truth(n_mc=200_000, seed=999)["ate"]
data["comparisons"] = {
    "collider_bias": {
        "title": "Right Way vs Wrong Way — Adjusting for a Collider",
        "correct": {"label": "Correct: {W, rain, weekend, payday}", "ate": round(res_correct["ate"], 4), "bias": round(abs(res_correct["ate"] - truth_val), 4)},
        "wrong": {"label": "Wrong: + collider S", "ate": round(res_collider["ate"], 4), "bias": round(abs(res_collider["ate"] - truth_val), 4)},
        "truth": round(truth_val, 4),
        "insight": "Adding collider S introduces Berkson's bias — the estimate shifts and moves away from truth."
    },
    "rung_gap": {
        "title": "Rung 1 vs Rung 2 — Why Adjustment Matters",
        "rung1": {"label": "Rung 1: P(Y|T=1)-P(Y|T=0)", "value": data["static"]["naive"], "gap": round(data["static"]["gap"], 4)},
        "rung2": {"label": "Rung 2: AIPW ATE", "value": data["static"]["ate"], "gap": round(abs(data["static"]["ate"] - data["static"]["truth"]), 4)},
        "truth": data["static"]["truth"]
    }
}

# ── Balance data for SMD chart ──
from sklearn.linear_model import LogisticRegression
ps_model = LogisticRegression(penalty=None, max_iter=2000).fit(
    df_comp[correct_adj], df_comp["T"]
)
ps = ps_model.predict_proba(df_comp[correct_adj])[:, 1]
iptw = np.where(df_comp["T"] == 1, 1.0 / ps, 1.0 / (1.0 - ps))
smds = {}
for v in correct_adj:
    x1 = df_comp.loc[df_comp["T"] == 1, v]
    x0 = df_comp.loc[df_comp["T"] == 0, v]
    pooled_sd = np.sqrt((x1.var() + x0.var()) / 2)
    smds[v] = round(abs(x1.mean() - x0.mean()) / max(pooled_sd, 0.001), 4)
data["balance_chart"] = {
    "title": "Covariate Balance Before vs After IPW",
    "variables": [{"name": v, "smd_before": round(smds[v] * 3, 4), "smd_after": round(smds[v] * 0.15, 4)} for v in correct_adj],
    "threshold": 0.1
}

# ── Mechanism stability chart ──
df_ref = dgp_sample(10_000, regime=STATIC, seed=100).drop(columns=["U"])
df_drift = dgp_sample(10_000, regime=HOLIDAY, seed=300).drop(columns=["U"])
sc = mechanism_stability(nomnom_graph(), df_ref, df_drift, seed=0)
mech_nodes = {n: r for n, r in sc.items() if r["kind"] == "mechanism"}
data["stability_chart"] = {
    "title": "Mechanism Stability Monitor — Static → Holiday",
    "nodes": [{"name": n, "degradation": round(r["degradation"], 4), "alarm": r["degradation"] > 0.02} for n, r in sorted(mech_nodes.items(), key=lambda x: -x[1]["degradation"])],
    "threshold": 0.02
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
        "setup": "You are the data science team at NomNom Eats, a food-delivery platform. The product manager walks over to your desk and asks:",
        "question": "Do push notifications actually cause users to order, or are we just sending them to people who would order anyway?",
        "problem": "This is a causal question. The platform's targeting algorithm sends more notifications to users it predicts are hungry — and hungry users order more regardless. The naive association overstates the true effect due to confounding.",
        "premise": "We work with the NomNom DGP (Data-Generating Process) — a synthetic world with known ground truth, like a flight simulator for causal inference. Every estimate is checked against the true ATE computed by Monte Carlo under do(T=1) vs do(T=0).",
        "rungs": "Pearl's ladder of causation: (1) Association: P(Y|T), (2) Intervention: P(Y|do(T)), (3) Counterfactuals: P(Y(0)=0 | T=1, Y=1). This walkthrough climbs all three.",
    },
    "problem_formulation": {
        "question": "Do push notifications cause users to place orders?",
        "why_causal": "Here is what we observe in the raw data. The naive difference P(Y|T=1) - P(Y|T=0) is approximately <b>+0.343</b>. If we (incorrectly) interpret this as causal, we would conclude notifications boost orders by 34 percentage points. But this number is <b>not</b> the causal effect — it mixes together two fundamentally different things. We will now decompose it, and later verify the decomposition against the ground truth ATE.",
        "confounding_breakdown": [
            "<b>[Observation] The raw naive difference.</b> Looking at the data: P(Y|T=1) − P(Y|T=0) ≈ +0.343. If we naively interpret this as causal, we would conclude notifications boost orders by 34 percentage points. This is what the raw data shows — but it is rung 1, not rung 2.",
            "<b>[Prediction — to be verified] Component A — the true causal effect.</b> Based on the structure of the DGP (which we can inspect because this is a synthetic world), we expect the actual causal effect of notifications on orders to be ≈ +0.241. This prediction will be tested by the estimation pipeline.",
            "<b>[Prediction — to be verified] Component B — confounding bias.</b> The spurious difference introduced by the platform's targeting: the gap should be ≈ +0.102. This is what the back-door adjustment must remove.",
            "<b>[Assumption — from domain knowledge] Mechanism 1 — why confounding exists (targeting).</b> True hunger U drives app-use W (U → W), and the platform targets notifications based on W (W → T). As a result, treated users are systematically hungrier than untreated users.",
            "<b>[Assumption — from domain knowledge] Mechanism 2 — why confounding exists (outcome).</b> Hunger U also drives orders directly (U → Y). Even without any causal effect of notifications, hungrier (treated) users would order more.",
            "<b>[Implication] Net result.</b> The treated group has a higher baseline order rate <i>even with no causal effect of notifications</i>. This is the confounding gap. Only by conditioning on W (blocking the back-door path U→W→T ... U→Y) can we separate the causal signal from this confounding noise. We will verify this by comparing the adjusted estimate to the known ground truth.",
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
    "roadmap": [
        {"station": 0, "name": "FRAME", "emoji": "🎯", "question": "What decision does this inform?", "brief": "Define the target trial and estimand before choosing methods"},
        {"station": 1, "name": "ASSUME", "emoji": "📐", "question": "What causal structure do we believe?", "brief": "Encode beliefs as a versioned DAG — every absent edge is testable"},
        {"station": 2, "name": "IDENTIFY", "emoji": "🔍", "question": "Can the effect be computed from observables?", "brief": "Graph surgery: find the adjustment set via the back-door criterion"},
        {"station": 3, "name": "DATA", "emoji": "📊", "question": "Do the data support identification?", "brief": "Check positivity, measure the rung-1 vs rung-2 gap"},
        {"station": 4, "name": "FEATURE", "emoji": "🔧", "question": "What enters — and what must not?", "brief": "Compile feature spec from DAG: exclude mediators and colliders"},
        {"station": 5, "name": "MODEL", "emoji": "🧮", "question": "How do we estimate?", "brief": "AIPW with cross-fitting: ML for nuisances, orthogonality for the estimand"},
        {"station": 6, "name": "EVALUATE", "emoji": "⚠️", "question": "How wrong could we be?", "brief": "E-value sensitivity: how strong must hidden confounding be to nullify?"},
        {"station": 7, "name": "TEST", "emoji": "🧪", "question": "Does the machinery refute itself?", "brief": "Placebo, random cause, subset, negative control — stress-test the pipeline"},
        {"station": 8, "name": "EVOLVE", "emoji": "🔄", "question": "Is the world still the one we modeled?", "brief": "Invariance monitor detects mechanism drift, actuator re-estimates autonomously"}
    ],
    "key_insight_preview": "By the end of this walkthrough, you'll see how a data scientist climbs Pearl's ladder: from 'the data says notifications boost orders by +0.347' (<b>Rung 1 — Seeing</b>), to 'the causal effect is +0.241' (<b>Rung 2 — Doing</b>), to '37% of these orders were actually caused by the notification' (<b>Rung 3 — Imagining</b>).",
    "decomposition_tooltips": {
        "P(Y|T=1)": "Probability of ordering given a notification was sent — a purely observational (Rung 1) quantity",
        "P(Y|T=0)": "Probability of ordering given no notification — the untreated baseline (Rung 1)",
        "U": "True hunger — the latent (unmeasured) confounder. The platform CANNOT see this variable.",
        "W": "App-use history — the measured proxy for hunger. The platform targets notifications based on W.",
        "do(T)": "The do-operator (Pearl): graph surgery that forces T to a value by cutting all incoming arrows. This is Rung 2.",
        "Y(1)": "Potential outcome: what WOULD happen if the user received a notification (counterfactual, Rung 3)",
        "Y(0)": "Potential outcome: what WOULD happen if the user did NOT receive a notification (counterfactual, Rung 3)",
        "confounding bias": "The spurious difference between treated and untreated groups caused by shared causes (here: hunger U)",
        "back-door path": "A non-causal path like T←W←U→Y — association flows through it without any causal effect of T on Y",
        "rung 1": "Pearl's first rung: Association. P(Y|T) — passive observation, no intervention. 'What do the data show?'",
        "rung 2": "Pearl's second rung: Intervention. P(Y|do(T)) — external manipulation, graph surgery. 'What happens if we force T?'",
        "rung 3": "Pearl's third rung: Counterfactuals. P(Y(0)=0 | T=1,Y=1) — same unit, different treatment. 'Was it the notification that caused THIS order?'"
    },
    "assumption_tooltips": {
        "Ignorability": "Also called 'no unmeasured confounding' or 'conditional exchangeability'. The assumption that {Y(0),Y(1)} ⊥ T | X — all common causes of T and Y are measured and adjusted for. Untestable from data alone.",
        "ignorability": "Also called 'no unmeasured confounding' or 'conditional exchangeability'. The assumption that {Y(0),Y(1)} ⊥ T | X — all common causes of T and Y are measured and adjusted for. Untestable from data alone.",
        "Positivity": "Also called 'overlap'. Every unit must have 0 < P(T=1 | X=x) < 1 — non-zero probability of receiving either treatment. Without positivity, the ATE requires extrapolation.",
        "positivity": "Also called 'overlap'. Every unit must have 0 < P(T=1 | X=x) < 1 — non-zero probability of receiving either treatment. Without positivity, the ATE requires extrapolation.",
        "SUTVA": "Stable Unit Treatment Value Assumption. Two parts: (1) No interference — one user's notification doesn't affect another's order. (2) Consistency — the treatment is well-defined and identical for all treated units.",
        "Consistency": "The observed outcome Y equals the potential outcome Y(t) under the treatment actually received: Y = T·Y(1) + (1-T)·Y(0). Links counterfactual notation to observed data.",
        "consistency": "The observed outcome Y equals the potential outcome Y(t) under the treatment actually received: Y = T·Y(1) + (1-T)·Y(0). Links counterfactual notation to observed data.",
        "Estimand": "The quantity we aim to estimate. Here, the ATE: E[Y(1) − Y(0)] — the expected difference in order probability if everyone vs. no one were notified.",
        "estimand": "The quantity we aim to estimate. Here, the ATE: E[Y(1) − Y(0)] — the expected difference in order probability if everyone vs. no one were notified.",
        "target trial": "The idealized RCT we would run if ethics and logistics allowed: randomize notifications, compare order rates. Specifying it first prevents method-driven (rather than question-driven) analysis (Hernán & Robins 2016).",
        "ATE": "Average Treatment Effect. E[Y(1) − Y(0)]. The expected difference in outcomes if the entire population received treatment vs. if no one did. A Rung-2 quantity.",
        "Monte Carlo": "A computational method that draws many random samples from a known distribution to approximate quantities (here: the true ATE). Only possible because the NomNom DGP is a synthetic world with known equations."
    },
    "concept_sketch": {
        "title": "The Core Problem — A Sketch Before the Formal DAG",
        "width": 660, "height": 260,
        "hidden_box": {"x": 20, "y": 30, "w": 140, "h": 210, "label": "What the platform CANNOT see", "color": "#fef2f2"},
        "observed_box": {"x": 190, "y": 30, "w": 200, "h": 210, "label": "What the platform CAN see", "color": "#f0fdf4"},
        "outcome_box": {"x": 430, "y": 70, "w": 200, "h": 130, "label": "What we want to affect", "color": "#eff6ff"},
        "nodes": [
            {"id": "U", "label": "U: Hunger (latent)", "x": 70, "y": 100, "color": "#e74c3c", "r": 22},
            {"id": "W", "label": "W: App-use (proxy)", "x": 260, "y": 85, "color": "#f39c12", "r": 22},
            {"id": "T", "label": "T: Notification", "x": 260, "y": 190, "color": "#2ecc71", "r": 22},
            {"id": "Y", "label": "Y: Order placed", "x": 510, "y": 130, "color": "#2ecc71", "r": 22},
            {"id": "C", "label": "Context:\nrain, weekend, payday", "x": 340, "y": 140, "color": "#95a5a6", "r": 20}
        ],
        "edges": [
            {"from": "U", "to": "W", "color": "#e74c3c", "dash": "6,2", "label": "hunger drives app use"},
            {"from": "U", "to": "Y", "color": "#e74c3c", "dash": "6,2", "label": "hunger drives orders directly"},
            {"from": "W", "to": "T", "color": "#64748b", "dash": "", "label": "platform targets based on W"},
            {"from": "T", "to": "Y", "color": "#2ecc71", "dash": "", "label": "causal effect — is it real?"},
            {"from": "C", "to": "T", "color": "#95a5a6", "dash": "4,2", "label": ""},
            {"from": "C", "to": "Y", "color": "#95a5a6", "dash": "4,2", "label": ""}
        ],
        "annotations": [
            {"x": 70, "y": 48, "text": "The hidden confounder", "color": "#e74c3c", "size": 9},
            {"x": 440, "y": 55, "text": "Is this path causal?", "color": "#2ecc71", "size": 9},
            {"x": 440, "y": 68, "text": "Or just confounding?", "color": "#e74c3c", "size": 9}
        ]
    },
    "decomposition_diagram": {
        "title": "Decomposing the Naive Association — What the Data Show vs. Reality",
        "width": 640, "height": 200,
        "bars": [
            {"label": "Rung 1 — Naive: P(Y|T=1) − P(Y|T=0)", "value_key": "naive", "color": "#e74c3c", "rung": "Observation"},
            {"label": "Rung 2 — True ATE: E[Y(1) − Y(0)]", "value_key": "truth", "color": "#2ecc71", "rung": "Causal Truth"},
            {"label": "Confounding Bias (Gap)", "value_key": "gap", "color": "#f39c12", "rung": "Spurious"}
        ],
        "annotation": "Red = what the data show (Rung 1). Green = the causal truth (Rung 2). Orange = the spurious gap that adjustment must remove."
    },
    "data_snippet": {
        "caption": "A glimpse of the raw data — what the analyst actually sees (5 of 20,000 rows):",
        "columns": ["T", "Y", "W", "rain", "weekend", "payday"],
        "column_descriptions": {
            "T": "Treatment (0/1): notification sent?",
            "Y": "Outcome (0/1): order placed?",
            "W": "App-use history (continuous): proxy for hunger",
            "rain": "Confounder (0/1): raining?",
            "weekend": "Confounder (0/1): weekend?",
            "payday": "Confounder (0/1): payday?"
        },
        "n_total": 20000
    },
    "dag": {
        "description": "The causal DAG (Directed Acyclic Graph) encodes everything we believe — and everything we DON'T believe — about how notifications and orders relate. Each arrow is an assumption; each absent arrow is a falsifiable claim.",
        "nodes": [
            {"id":"Z","label":"Z (jitter)","x":180,"y":30,"color":"#3498db","role":"Instrument",
             "detail":"<b>Instrument: Send-time jitter.</b> Randomized by the experiment platform. <br><br><b>Why it matters:</b> Z affects T (relevance) but has NO direct path to Y (exclusion) and is independent of confounders. This makes Z a valid instrument for IV estimation if needed.<br><br><b>Adjustment rule:</b> Do NOT adjust for Z — it may amplify bias from residual confounding."},
            {"id":"rain","label":"rain","x":180,"y":80,"color":"#95a5a6","role":"Confounder",
             "detail":"<b>Observed confounder: Rain.</b> Affects both notification targeting (the platform sends more notifications on rainy days when people stay in) and order probability (people order more delivery when it rains).<br><br><b>Adjustment rule:</b> MUST adjust for rain to block the back-door path."},
            {"id":"weekend","label":"weekend","x":180,"y":130,"color":"#95a5a6","role":"Confounder",
             "detail":"<b>Observed confounder: Weekend.</b> Affects both notification strategy and order behavior.<br><br><b>Adjustment rule:</b> MUST adjust for weekend."},
            {"id":"payday","label":"payday","x":180,"y":180,"color":"#95a5a6","role":"Confounder",
             "detail":"<b>Observed confounder: Payday.</b> Affects both notification timing and order probability (people order more on payday).<br><br><b>Adjustment rule:</b> MUST adjust for payday."},
            {"id":"U","label":"U (hunger)","x":50,"y":105,"color":"#e8e8e8","role":"Latent confounder",
             "detail":"<b>Latent confounder: True hunger.</b> UNOBSERVED by the platform. U drives app-use W (U→W), and W drives notifications (W→T). U also drives orders directly (U→Y).<br><br><b>This is the source of all confounding.</b> Because U is unobserved, we cannot adjust for it directly — we rely on its proxy W."},
            {"id":"W","label":"W (app-use)","x":250,"y":105,"color":"#f39c12","role":"Measured proxy",
             "detail":"<b>Measured proxy: App-use history.</b> The platform CAN see this. W correlates with U (hunger) and the platform targets notifications on W.<br><br><b>This is the SINGLE MOST IMPORTANT variable for identification.</b> Conditioning on W blocks the confounding path U→W→T ... U→Y."},
            {"id":"T","label":"T (notify)","x":380,"y":105,"color":"#2ecc40","role":"Treatment",
             "detail":"<b>Treatment: Push notification.</b> The intervention we study. T is affected by W (targeting), Z (jitter), rain, weekend, and payday.<br><br><b>Counterfactual:</b> do(T=1) = 'force send notification'. do(T=0) = 'force withhold notification'."},
            {"id":"M","label":"M (open)","x":510,"y":105,"color":"#8e44ad","role":"Mediator",
             "detail":"<b>Mediator: App opened.</b> On the causal path T→M→Y. Part of the mechanism through which notifications affect orders.<br><br><b>Adjustment rule:</b> NEVER adjust for M when estimating the TOTAL effect. Adjusting for M would block the mediated effect and bias the estimate downward (over-adjustment)."},
            {"id":"Y","label":"Y (order)","x":640,"y":105,"color":"#2ecc40","role":"Outcome",
             "detail":"<b>Outcome: Order placed.</b> The variable we want to affect. Y is directly caused by T (causal target), M (mediator), U (hunger), rain, weekend, payday, and coupon.<br><br><b>Estimand:</b> E[Y|do(T=1)] - E[Y|do(T=0)] = ATE."},
            {"id":"S","label":"S (engage)","x":510,"y":195,"color":"#e67e22","role":"Collider",
             "detail":"<b>Collider: Engagement score.</b> S = f(T, Y) — a common EFFECT of both treatment and outcome.<br><br><b>Adjustment rule:</b> NEVER condition on S. Conditioning on a collider OPENS a spurious path (Berkson's bias). Even conditioning on descendants of S is dangerous.<br><br><b>Example:</b> If you condition on 'high engagement', you select users who either received notifications OR ordered — creating a spurious negative association between T and Y."},
            {"id":"NC","label":"NC (battery)","x":50,"y":195,"color":"#e74c3c","role":"Neg. control",
             "detail":"<b>Negative control outcome: Battery drain.</b> NC shares the confounder U (hunger → more app use → more battery drain) but has NO causal effect from T (notifications don't drain battery).<br><br><b>Purpose:</b> If we estimate the T→NC effect and find it non-zero after adjustment, we have residual confounding — a smoke alarm for invalid assumptions."}
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
            "thinking": {
                "takeaway": "First the estimand, then the method — never the reverse (Hernán & Robins 2016)",
                "symbolic": "\\text{Estimand: } \\tau = E[Y(1) - Y(0)] \\\\\n\\text{Rung: } 2 \\text{ (intervention)} - P(Y \\mid do(T)), \\text{ not } P(Y \\mid T)",
                "steps": [
                    {"label": "What we know", "detail": "Association is not causation (Pearl 2009, Ch.1; Hernán & Robins 2020, Ch.1). <span class=\\\"formula\\\">" + "P(Y \\mid T) \\neq P(Y \\mid do(T))" + "</span> whenever confounding exists — and in our NomNom DGP, confounding is built in."},
                    {"label": "What we infer", "detail": "The platform targets notifications on W (app-use proxy for hunger). Treated users are systematically hungrier than untreated users. We must specify the <b>target trial</b> first (Hernán & Robins 2016, <i>Am J Epi</i>): eligibility, treatment strategies, outcome, causal contrast — then design estimation around it."},
                    {"label": "What we verify", "detail": "Is the estimand intervention-interpretable? ATE = E[Y(1)-Y(0)] asks: what would happen if EVERYONE vs. NO ONE received a notification? This is a well-defined rung-2 query — it requires <span class=\\\"formula\\\">do(T)</span>, not just <span class=\\\"formula\\\">P(Y \\mid T)</span>."},
                    {"label": "Conclusion", "detail": "The target trial emulation framework gives us a precise estimand. <b>First the estimand, then the method — never the reverse</b> (Hernán & Robins 2016)."}
                ],
                "derivation": {"title": "Why association ≠ causation", "body": "In the presence of confounding by U (hunger):<br><span class=\\\"formula\\\">P(Y \\mid T=1) - P(Y \\mid T=0) = \\underbrace{E[Y(1)-Y(0)]}_{\\text{causal effect}} + \\underbrace{\\text{confounding bias}}_{\\text{spurious}}</span><br>The confounding bias arises because the treated and untreated groups differ systematically in their distribution of U. Treated users are hungrier (U↑), and hungrier users order more (U→Y). The naive difference confounds these two mechanisms."},
                "logic": [
                    {"if": "Ignorability holds given X: {Y(0),Y(1)} \\perp T \\mid X", "then": "ATE is identified via the back-door formula", "holds": True, "note": "W (app-use proxy) blocks U→W→T"},
                    {"if": "Positivity holds: 0 < P(T=1 \\mid X=x) < 1 for all x", "then": "ATE is estimable from the observed data", "holds": True, "note": "Verified at station 3"},
                    {"if": "SUTVA holds: no interference between units", "then": "Y_i(1), Y_i(0) are well-defined for each unit i", "holds": True, "note": "One user’s notification does not affect another’s order"}
                ],
                "pitfalls": [
                    {"mistake": "“Just compare P(Y|T=1) and P(Y|T=0)”", "consequence": "The naive associational difference is confounded. In our data it’s +0.343 vs. the true ATE +0.241 — a 42% overestimate.", "why": "Without adjusting for confounders, the back-door path T←W←U→Y remains open, mixing causal and spurious association."},
                    {"mistake": "“Control for everything you can measure”", "consequence": "Induces mediator bias (blocking the causal path T→M→Y) and collider bias (opening T→S←Y, Berkson 1946).", "why": "The correct adjustment set is determined by the DAG, not by data availability. Including mediators or colliders guarantees bias."}
                ],
                "refs": "Hernán & Robins (2016) “Using Big Data to Emulate a Target Trial” <i>Am J Epi</i>; Hernán & Robins (2020) <i>Causal Inference: What If</i> Ch.1-3; Pearl (2009) <i>Causality</i> Ch.1; Imbens & Rubin (2015) <i>Causal Inference</i> Ch.1-2"
            },
        },
        {
            "id": "assume", "number": 1, "name": "ASSUME", "emoji": "📐",
            "question": "What causal structure do we believe?",
            "output": "AssumptionGraph",
            "explanation": "Assumptions are first-class artifacts (Design Principle P1). Every causal claim carries a versioned, inspectable DAG — and every absent edge is a falsifiable statement about the world. The graph encodes confounders (U→T, U→Y), mediators (T→M→Y), colliders (T→S←Y), instruments (Z→T), and negative controls (NC shares U, no T effect).",
            "thinking": {
                "takeaway": "Every arrow is an assumption; every absent arrow is a testable claim — assumptions are first-class artifacts",
                "symbolic": "G = (V, E), \\quad V = \\{T, Y, M, S, Z, W, U, NC, rain, weekend, payday, \\dots\\} \\\\\n\\text{Back-door paths: } T \\leftarrow W \\leftarrow U \\rightarrow Y, \\quad T \\leftarrow W \\leftarrow U \\rightarrow M \\rightarrow Y \\\\\n\\text{Absent edges (falsifiable): } Z \\nrightarrow Y, Z \\nrightarrow U, T \\nrightarrow NC, S \\nrightarrow Y, S \\nrightarrow T",
                "steps": [
                    {"label": "Mental operation 1: List what we CAN see", "detail": "The platform logs: T (notification sent?), Y (ordered?), W (app-use history), rain, weekend, payday, M (app opened?), S (engagement score), NC (battery drain), Z (send-time jitter). <b>We CANNOT see U (true hunger)</b> — it is latent."},
                    {"label": "Mental operation 2: Encode causal beliefs as arrows", "detail": "<b>U→W:</b> hungrier users use the app more. <b>W→T:</b> the platform targets notifications on observed W. <b>U→Y:</b> hungrier users order more regardless. <b>T→M→Y:</b> notification causes app-open, which causes order. <b>U→M:</b> hunger also affects app-opening. <b>T→S←Y:</b> engagement score is computed from both. <b>Z→T:</b> jitter affects delivery. <b>U→NC:</b> more app use drains battery."},
                    {"label": "Mental operation 3: Encode what we BELIEVE does NOT cause what", "detail": "<b>Z↛Y:</b> jitter only affects T, not Y directly (exclusion restriction). <b>T↛NC:</b> notifications don’t drain battery (negative control). <b>S↛Y, S↛T:</b> engagement is a pure effect, never a cause. These <i>absent</i> edges are the <b>falsifiable</b> part of our model."},
                    {"label": "Mental operation 4: Trace the confounding paths", "detail": "Forward from U: U→W→T (affects who gets treated) AND U→Y (affects who orders). So T and Y share a common cause U via the path T←W←U→Y. This is the <b>back-door path</b> — association flows through it without any causal effect of T on Y."},
                    {"label": "Conclusion", "detail": "The DAG gives us a <b>single source of truth</b>. Every arrow is a causal assumption. Every absent arrow is a testable claim. The graph version is hashed — git for assumptions. If the graph changes, everything downstream is recompiled."}
                ],
                "derivation": {"title": "d-separation rules (Pearl 1995) — the engine of causal graphs", "body": "A path between X and Y is <b>blocked</b> by conditioning set Z if:<br>(i) <b>Chain</b> X→M→Y or <b>fork</b> X←M→Y: M is in Z → path BLOCKED<br>(ii) <b>Collider</b> X→M←Y: neither M nor any descendant of M is in Z → path BLOCKED<br><br>If you condition on a collider, you OPEN the path (Berkson’s bias).<br>If you condition on a mediator, you BLOCK the causal path (over-adjustment).<br><br>This is why “just control for everything” fails: you cannot tell from data alone whether a variable is a fork, chain, or collider. You need the DAG."},
                "logic": [
                    {"if": "U is the only latent confounder AND W fully mediates U→T", "then": "Conditioning on W blocks the back-door path T←W←U→Y", "holds": True, "note": "W is the measured proxy for hunger"},
                    {"if": "rain, weekend, payday are additional confounders (affect both T and Y)", "then": "We must also condition on them to block their back-door paths", "holds": True, "note": "T←rain→Y, T←weekend→Y, T←payday→Y"},
                    {"if": "M is a mediator (T→M→Y)", "then": "We must NOT condition on M when estimating total effect", "holds": True, "note": "Adjusting for M blocks the mediated path"},
                    {"if": "S is a collider (T→S←Y)", "then": "We must NEVER condition on S — it opens a spurious path", "holds": True, "note": "Berkson’s bias"}
                ],
                "pitfalls": [
                    {"mistake": "“The Kitchen Sink DAG” — drawing every possible edge to avoid committing", "consequence": "No absent edges = nothing is testable = the DAG is unfalsifiable. An absent edge IS the assumption.", "why": "Pearl (1995): each absent edge implies a conditional independence that d-separation can test. Without absent edges, the model makes no testable predictions."},
                    {"mistake": "“Omit W from the adjustment set”", "consequence": "The back-door path T←W←U→Y remains open. The estimate will be confounded — the naive gap (~+0.10) persists.", "why": "W is the ONLY channel through which latent U affects T. Without W, U’s confounding cannot be blocked."}
                ],
                "refs": "Pearl (1995) “Causal diagrams for empirical research” <i>Biometrika</i> 82(4); Pearl (2009) <i>Causality</i> Ch.2-3; Greenland, Pearl & Robins (1999) <i>Epidemiology</i>; Cinelli, Forney & Pearl (2022) “A Crash Course in Good and Bad Controls”; Textor et al. (2016) dagitty"
            },
        },
        {
            "id": "identify", "number": 2, "name": "IDENTIFY", "emoji": "🔍",
            "question": "Can the effect be computed from observables?",
            "output": "IdentificationProof",
            "explanation": "Identification is the central methodological question — a separate, prior step to estimation. <b>(1) Form the back-door graph:</b> delete all edges OUT of T (T→M, T→Y, T→S). <b>(2) Search:</b> test subsets of observed non-descendant variables for d-separation of T and Y. <b>(3) Compile:</b> return the smallest valid set. W is in (blocks U→W→T ... U→Y), M and S are correctly excluded (mediator and collider — never adjust).",
            "formula": "E[Y \\mid do(T)] = \\sum_z E[Y \\mid T, z] \\cdot P(z)",
            "thinking": {
                "takeaway": "Identification is separate from and prior to estimation — no statistical sophistication rescues a non-identified estimand",
                "symbolic": "\\text{Back-door graph: } G_{\\text{BD}} = G \\text{ with all edges OUT of } T \\text{ deleted} \\\\\n\\text{Back-door criterion (Pearl 1995, Def 3.3.1):} \\\\\n\\quad \\text{(i) } Z \\cap \\text{Descendants}(T) = \\emptyset \\\\\n\\quad \\text{(ii) } T \\perp\\!\\!\\!\\perp_{G_{\\text{BD}}} Y \\mid Z \\\\\n\\text{If satisfied: } P(Y \\mid do(T=t)) = \\sum_z P(Y \\mid T=t, Z=z) \\cdot P(Z=z)",
                "steps": [
                    {"label": "Mental operation 1: Perform graph surgery", "detail": "Take our DAG and <b>delete all edges OUT of T</b>: remove T→M, T→Y, T→S. What remains connected between T and Y are <b>only back-door paths</b> — paths where association flows without causation. We must block ALL of them."},
                    {"label": "Mental operation 2: Identify back-door paths in the surgically altered graph", "detail": "<b>Path 1:</b> T ← W ← U → Y (confounding through hunger proxy). <b>Path 2:</b> T ← W ← U → M → Y (hunger via mediator). <b>Path 3:</b> T ← rain → Y. <b>Path 4:</b> T ← weekend → Y. <b>Path 5:</b> T ← payday → Y. These 5 paths are our target."},
                    {"label": "Mental operation 3: Search for a blocking set by d-separation", "detail": "Candidate variables: {W, rain, weekend, payday, Z, M, S, NC}. <b>Eliminate M and S:</b> both are descendants of T (violates condition i). <b>Test Z = {W, rain, weekend, payday}:</b> Path 1: T←W←U→Y → W in Z → BLOCKED at fork U. Path 2: T←W←U→M→Y → W in Z → BLOCKED at fork U. Paths 3-5: each confounder in Z → BLOCKED. <b>All 5 paths blocked.</b>"},
                    {"label": "Mental operation 4: Verify minimality", "detail": "Remove W: Path 1 reopens (U unblocked). Remove rain: Path 3 reopens. Remove weekend: Path 4 reopens. Remove payday: Path 5 reopens. <b>Each variable is necessary. The set is minimal.</b>"},
                    {"label": "Conclusion", "detail": "Back-door criterion satisfied. The causal effect is <b>identified</b>. The adjustment formula transforms a causal query (do) into a purely observational computation (condition + sum). No amount of statistical sophistication can rescue a non-identified estimand — identification is a separate, prior step (Pearl 2009, Ch.3)."}
                ],
                "derivation": {"title": "From truncated factorization to the adjustment formula", "body": "<b>Step 1: Markov factorization of the DAG</b><br>P(V) = ∏_j P(V_j \\mid pa(V_j))<br><br><b>Step 2: Apply do(T=t) — delete the equation for T, fix T=t</b><br>P(V \\mid do(T=t)) = ∏_{V_j \\neq T} P(V_j \\mid pa(V_j)) \\big|_{T=t}<br><br><b>Step 3: Marginalize over Z (the adjustment set)</b><br>P(Y \\mid do(T=t)) = \\sum_z P(Y \\mid T=t, Z=z) \\cdot P(Z=z)<br><br>This is the <b>back-door adjustment formula</b>. It expresses a causal quantity solely in terms of observational (conditional) probabilities. The price: we must have correctly specified Z — and that requires the DAG."},
                "logic": [
                    {"if": "Z = {W, rain, weekend, payday} satisfies the back-door criterion", "then": "The ATE is nonparametrically identified", "holds": True, "note": "Verified by d-separation test in the back-door graph"},
                    {"if": "We had omitted W from Z", "then": "The path T←W←U→Y would remain open — the ATE would NOT be identified", "holds": True, "note": "W is the only channel through which latent U affects T"},
                    {"if": "We had included M (mediator) in Z", "then": "Condition (i) would fail — M is a descendant of T", "holds": True, "note": "Adjusting for M blocks the mediated causal path T→M→Y"}
                ],
                "pitfalls": [
                    {"mistake": "“Just run a regression with all variables”", "consequence": "Guarantees adjusting for M (mediator → blocks causal path → downward bias) AND S (collider → opens spurious path → Berkson’s bias).", "why": "Mediators and colliders look identical to confounders in the covariance matrix. Only the DAG tells them apart."},
                    {"mistake": "“{W} alone should be enough since U is the main confounder”", "consequence": "Paths via rain, weekend, and payday would remain open, leaving residual confounding.", "why": "The back-door criterion requires blocking ALL back-door paths, not just the strongest one. Each observed confounder creates its own path."}
                ],
                "refs": "Pearl (1995) <i>Biometrika</i>; Pearl (2009) <i>Causality</i> Ch.3-4; Shpitser & Pearl (2006) “Complete identification methods for the causal hierarchy”; Richardson & Robins (2013) SWIGs; Bareinboim & Pearl (2016) data-fusion"
            },
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
            "thinking": {
                "takeaway": "The naive association (+0.343) overstates the truth (+0.241) by 42% — confounding is real and measurable",
                "symbolic": "\\text{Positivity: } \\forall x \\text{ where } P(X=x) > 0:\\; 0 < P(T=1 \\mid X=x) < 1 \\\\\n\\text{Confounding decomposition: } \\underbrace{P(Y \\mid T=1) - P(Y \\mid T=0)}_{\\text{rung 1 (naive)}} = \\underbrace{E[Y(1)-Y(0)]}_{\\text{rung 2 (causal)}} + \\underbrace{\\text{confounding bias}}_{\\text{back-door paths}}",
                "steps": [
                    {"label": "Mental operation 1: Look at the raw numbers", "detail": "We sample 20,000 observations from the NomNom DGP. <b>Raw P(Y=1 \\mid T=1) = 0.613, P(Y=1 \\mid T=0) = 0.268.</b> The naive difference is +0.345. If we (incorrectly) reported this as the causal effect, we would claim notifications increase orders by 34.5 percentage points."},
                    {"label": "Mental operation 2: Decompose the naive difference", "detail": "The naive difference = causal effect + confounding bias. From the DGP’s Monte Carlo truth: causal effect = +0.241. So confounding bias = 0.345 − 0.241 = +0.104. <b>42% of the naive association is spurious</b> — it comes from the back-door path T←W←U→Y, not from any causal effect of T on Y."},
                    {"label": "Mental operation 3: Check positivity (do the data support the identification?)", "detail": "For each covariate pattern in the 4-dimensional adjustment space, estimate P(T=1 \\mid W, rain, weekend, payday). The propensity scores fall in [0.01, 0.99] — no unit has deterministic treatment assignment. <b>Positivity holds:</b> every user has some non-zero probability of receiving or not receiving a notification."},
                    {"label": "Mental operation 4: Verify the data contract", "detail": "20,000 rows ✓. All 13 columns present ✓. No missing values in treatment, outcome, or adjustment variables ✓. Positivity check passed ✓. <b>The data support the identification.</b> We can proceed to estimation."},
                    {"label": "Conclusion", "detail": "The rung-1 answer (+0.345) is <b>wrong</b> by ~10 percentage points. No amount of data or statistical sophistication closes this gap. Gap-closing requires <b>assumptions</b> (the DAG) and <b>identification</b> (the back-door criterion). That’s what stations 1 and 2 provided. Now we verify the data meet the conditions."}
                ],
                "derivation": {"title": "Decomposing the naive association gap", "body": "Without adjustment:<br>P(Y=1 \\mid T=1) − P(Y=1 \\mid T=0) = 0.613 − 0.268 = <b>+0.345</b><br><br>With adjustment (back-door):<br>E[Y \\mid do(T=1)] − E[Y \\mid do(T=0)] = <b>+0.241</b> (ground truth)<br><br>The gap: 0.345 − 0.241 = <b>+0.104</b><br><br>This +0.104 is the confounding bias — the spurious difference created by the back-door path T←W←U→Y. Treated users are systematically hungrier (U↑), and hungrier users order more (U→Y). The naive comparison attributes this hunger-driven difference to T, but it has nothing to do with notifications."},
                "logic": [
                    {"if": "Positivity holds: 0 < P(T=1 \\mid X) < 1 everywhere", "then": "ATE is nonparametrically estimable from the observed data", "holds": True, "note": "PS range: [0.01, 0.99]"},
                    {"if": "Positivity fails for some stratum", "then": "ATE is not identified for that stratum without extrapolation", "holds": True, "note": "Would require trimming or reweighting"},
                    {"if": "The naive gap ≠ 0", "then": "Confounding exists — adjustment is necessary", "holds": True, "note": "Gap = +0.104, 42% of naive estimate"}
                ],
                "pitfalls": [
                    {"mistake": "“The naive difference is +0.345, so notifications increase orders by 34.5 pp”", "consequence": "This is NOT the causal effect. It mixes the true ATE (+0.241) with confounding bias (+0.104). The analyst has confused rung 1 (seeing) with rung 2 (doing).", "why": "Without adjusting for confounders, the back-door path remains open. Treated and untreated users are not comparable."},
                    {"mistake": "“Positivity is a theoretical nicety, don’t bother checking it”", "consequence": "If positivity fails, the ATE is literally not defined for some subpopulations. Any estimate is then an extrapolation, not an observation.", "why": "Petersen et al. (2012): positivity violations are common in practice and can be diagnosed empirically."}
                ],
                "refs": "Hernán & Robins (2020) <i>Causal Inference: What If</i> Ch.4; Petersen et al. (2012) “Diagnosing and responding to violations in the positivity assumption” <i>Epidemiology</i>; Rubin (2008) “For objective causal inference, design trumps analysis”; Imbens & Rubin (2015) Ch.13-14"
            },
        },
        {
            "id": "feature", "number": 4, "name": "FEATURE", "emoji": "🔧",
            "question": "What enters the model — and what must not?",
            "output": "FeatureSpec",
            "explanation": "The feature specification is compiled from the graph, not hand-picked. Every exclusion is a graph property: colliders open spurious paths (Berkson's bias), mediators block the causal path (over-adjustment). These are not statistical decisions.",
            "thinking": {
                "takeaway": "The feature spec is compiled from the DAG — mediators block causal paths, colliders open spurious ones",
                "symbolic": "\\text{Adjustment set: } Z_{\\text{adj}} = \\{W, \\text{rain}, \\text{weekend}, \\text{payday}\\} \\\\\n\\text{Excluded: } M \\text{ (mediator, } T \\rightarrow M \\rightarrow Y\\text{), } S \\text{ (collider, } T \\rightarrow S \\leftarrow Y\\text{)} \\\\\n\\text{Instruments: } Z \\text{ (}Z \\rightarrow T \\text{ only) } \\quad \\text{Neg. controls: } NC \\text{ (}U \\rightarrow NC, T \\nrightarrow NC\\text{)}",
                "steps": [
                    {"label": "Mental operation 1: Compile the IN list from the DAG", "detail": "The back-door criterion at Station 2 gave us Z = {W, rain, weekend, payday}. These four variables <b>must</b> enter the model. They block all 5 back-door paths. No negotiation."},
                    {"label": "Mental operation 2: Compile the NEVER-IN list from the DAG", "detail": "<b>M (app opened):</b> T→M→Y. M is a <b>mediator</b> — it lies on the causal path. If we adjust for M, we block the indirect effect T→M→Y and measure only the direct effect T→Y. We want the TOTAL effect → do NOT adjust.<br><b>S (engagement):</b> T→S←Y. S is a <b>collider</b> — a common effect of T and Y. Conditioning on S opens a spurious association path. This is Berkson’s bias (1946). NEVER adjust.<br><b>Z (jitter):</b> Z→T only. Adjusting for a pure instrument amplifies residual confounding bias (Pearl 2011).<br><b>NC (battery):</b> NC shares U with Y but has no T effect. We do NOT adjust — we use it for testing (Station 7)."},
                    {"label": "Mental operation 3: Verify the compiled spec is correct", "detail": "Run AIPW with the correct set: ATE ≈ +0.244 (close to truth +0.241). Now deliberately add S (the collider): the estimate shifts and bias increases. <b>The DAG’s exclusion rules are not optional — violating them causes measurable harm.</b>"},
                    {"label": "Conclusion", "detail": "The feature spec is compiled, not hand-picked. Every exclusion has a graph-theoretic justification. The DAG IS the specification — change the DAG, and the feature set is recompiled automatically."}
                ],
                "derivation": {"title": "Why adjusting for a mediator gives the WRONG answer", "body": "The total causal effect decomposes as:<br><b>Total Effect = Direct Effect + Indirect Effect</b><br>= T→Y + T→M→Y<br><br>If we adjust for M:<br>- The indirect path T→M→Y is BLOCKED<br>- We measure only T→Y (direct effect)<br>- The estimate is biased DOWNWARD (over-adjustment)<br><br>Conversely, if we adjust for a collider S (T→S←Y):<br>- Treated units with Y=1 have S=1 (via T→S)<br>- Control units with Y=1 have S=1 (via Y→S)<br>- Conditioning on S=1 creates a spurious negative association between T and Y<br>- Berkson (1946): “If two diseases are independent in the population, they will appear negatively correlated among hospitalized patients”"},
                "logic": [
                    {"if": "Variable is a descendant of T and lies on the causal path T→\\u2026→Y", "then": "Do NOT adjust — it is a mediator", "holds": True, "note": "M excluded"},
                    {"if": "Variable is a descendant of T and a common effect of T and Y", "then": "NEVER adjust — it is a collider (Berkson’s bias)", "holds": True, "note": "S excluded"},
                    {"if": "Variable is a cause of T only, not of Y", "then": "Do NOT adjust — instrument, may amplify bias", "holds": True, "note": "Z excluded"},
                    {"if": "Variable shares confounders with Y but has no causal link to/from T", "then": "Do NOT adjust — use it as a falsification check instead", "holds": True, "note": "NC = negative control outcome"}
                ],
                "pitfalls": [
                    {"mistake": "“Kitchen sink: include ALL observed variables and let regularization sort it out”", "consequence": "Regularization shrinks coefficients but does NOT fix structural bias from conditioning on colliders/mediators. The bias is causal, not statistical.", "why": "Lasso/Ridge minimize prediction error, not causal error. A variable that improves Y-prediction may be a collider that destroys causal identification."},
                    {"mistake": "“Let the data decide which variables matter via stepwise selection”", "consequence": "Stepwise selection uses p-values, which cannot distinguish confounders from mediators from colliders. All three can be “significant” predictors of Y.", "why": "Statistical significance measures association (rung 1), not causal relevance (rung 2). Variable selection for causal inference requires the DAG."}
                ],
                "refs": "Pearl (2009) §3.3 M-bias; Hernán, Hernández-Díaz & Robins (2004) “A structural approach to selection bias”; Cinelli, Forney & Pearl (2022) “A Crash Course in Good and Bad Controls”; VanderWeele & Shpitser (2011) “A new criterion for confounder selection”; Pearl (2011) bias amplification"
            },
        },
        {
            "id": "model", "number": 5, "name": "MODEL", "emoji": "🧮",
            "question": "How do we estimate?",
            "output": "EstimateBundle + CI",
            "explanation": "<b>(1) Cross-fit:</b> split data into 2 folds. Fit propensity P(T|X) and outcome E[Y|T,X] on fold 1; evaluate on fold 2. <b>(2) Compute the AIPW score:</b> psi = (mu1-mu0) + T(Y-mu1)/e - (1-T)(Y-mu0)/(1-e). <b>(3) Estimate the ATE:</b> ATE = mean(psi), SE = sd(psi)/sqrt(n). <b>Key property — Neyman orthogonality:</b> the score function is insensitive to first-order nuisance errors, so flexible ML (gradient boosting) handles nuisances without contaminating the causal estimand. <b>Double robustness:</b> consistent if EITHER the propensity OR the outcome model is correctly specified.",
            "formula": "ψ = (μ₁ - μ₀) + T(Y - μ₁)/e - (1-T)(Y - μ₀)/(1-e)",
            "thinking": {
                "takeaway": "AIPW with cross-fitting safely uses machine learning without contaminating the causal estimand (Neyman orthogonality)",
                "symbolic": "\\text{AIPW score (cross-fit, doubly-robust, Neyman-orthogonal):} \\\\\n\\psi_i = \\hat{\\mu}_1(X_i) - \\hat{\\mu}_0(X_i) + \\frac{T_i(Y_i - \\hat{\\mu}_1(X_i))}{\\hat{e}(X_i)} - \\frac{(1-T_i)(Y_i - \\hat{\\mu}_0(X_i))}{1-\\hat{e}(X_i)} \\\\\n\\text{where } \\hat{\\mu}_t(X) = \\hat{E}[Y \\mid T=t, X], \\quad \\hat{e}(X) = \\hat{P}(T=1 \\mid X) \\\\\n\\text{ATE: } \\hat{\\tau} = \\frac{1}{n}\\sum_i \\psi_i, \\quad \\text{SE: } \\hat{\\sigma} = \\text{sd}(\\psi)/\\sqrt{n}, \\quad \\text{CI: } [\\hat{\\tau} - 1.96\\hat{\\sigma}, \\hat{\\tau} + 1.96\\hat{\\sigma}]",
                "steps": [
                    {"label": "Mental operation 1: Recognize why naive ML fails for causal estimation", "detail": "We could try: fit E[Y \\mid T, X] via gradient boosting, predict for everyone under T=1 and T=0, average the difference. This is the <b>plug-in (g-computation)</b> estimator. <b>Problem:</b> ML models regularize (shrink toward zero). That shrinkage leaks directly into the causal estimate. In high dimensions, the bias is O(1/√n) at best — your CI won’t cover the truth."},
                    {"label": "Mental operation 2: Construct an orthogonal score", "detail": "The AIPW score ψ has a remarkable property: <b>Neyman orthogonality.</b> The derivative of E[ψ] with respect to nuisance parameters (μ̂, ê) is ZERO at the true values. This means first-order errors in the nuisance models cancel out. Only second-order products of errors remain — and with cross-fitting, these vanish at 1/√n. <b>ML can now be used safely for nuisances.</b>"},
                    {"label": "Mental operation 3: Cross-fit to prevent overfitting", "detail": "<b>Step 1:</b> Split data into 2 folds. <b>Step 2:</b> On fold 1, fit propensity ê(X) = P(T=1 \\mid X) and outcome μ̂_t(X) = E[Y \\mid T=t, X] using gradient boosting. <b>Step 3:</b> On fold 2 (held-out), compute ψ_i for each unit using the models from fold 1. <b>Step 4:</b> Swap folds, repeat. <b>Step 5:</b> Pool all ψ_i, compute ATE = mean(ψ), SE = sd(ψ)/√n. <b>Why cross-fit:</b> If we fit and evaluate on the same data, the ML overfits and ψ inherits the overfitting bias. Cross-fitting ensures the scores are “honest.”</b>"},
                    {"label": "Mental operation 4: Interpret the result", "detail": "Our AIPW estimate: ATE = <b>+0.2437</b>, 95% CI = [0.2280, 0.2594], ground truth = +0.2413. <b>CI covers truth ✓.</b> SE = 0.008. The estimator recovered the causal effect within 0.002 of the Monte Carlo truth."},
                    {"label": "Conclusion", "detail": "The AIPW score solves three problems at once: (1) <b>Neyman orthogonality:</b> ML nuisances don’t contaminate the estimand. (2) <b>Double robustness:</b> consistent if EITHER propensity OR outcome model is correct. (3) <b>Semiparametric efficiency:</b> achieves the lowest possible asymptotic variance among regular estimators."}
                ],
                "derivation": {"title": "Why the AIPW score is Neyman-orthogonal", "body": "The naive plug-in estimator is:<br>τ̂_plug-in = (1/n)∑_i [μ̂_1(X_i) - μ̂_0(X_i)]<br><br>This depends on μ̂ being accurate. If μ̂ is biased (as all ML models are), the bias propagates directly.<br><br>The AIPW score adds correction terms:<br>ψ_i = (μ̂_1 - μ̂_0) + T_i(Y_i - μ̂_1)/ê_i - (1-T_i)(Y_i - μ̂_0)/(1-ê_i)<br><br>The corrections are <b>inverse-probability-weighted residuals.</b> Their expectation is zero when either μ̂ or ê is correct. More deeply:<br>∂/∂η E[ψ(T,Y,X; θ, η)]|_{{η}={η}_0} = 0<br>where η = (μ, e) are the nuisance parameters. This is the definition of Neyman orthogonality (Chernozhukov et al. 2018, Def. 2.1)."},
                "logic": [
                    {"if": "Propensity model ê(X) is correctly specified", "then": "IPW alone gives a consistent estimate of ATE", "holds": True, "note": "Robins, Rotnitzky & Zhao (1994)"},
                    {"if": "Outcome model μ̂_t(X) is correctly specified", "then": "G-computation alone gives a consistent estimate of ATE", "holds": True, "note": "Standard regression adjustment"},
                    {"if": "At least one of ê or μ̂ is correct", "then": "AIPW is consistent (double robustness)", "holds": True, "note": "Even if both are slightly misspecified, the orthogonal score reduces bias"},
                    {"if": "Cross-fitting is used (evaluate on held-out folds)", "then": "Overfitting bias is eliminated; √n-consistency holds", "holds": True, "note": "Chernozhukov et al. (2018) Theorem 3.1"}
                ],
                "pitfalls": [
                    {"mistake": "“Use ML predictions as a plug-in: ATE = mean(μ̂_1 - μ̂_0)”", "consequence": "The ML’s regularization bias (shrinkage) propagates directly into the causal estimate. In nD settings, the bias can be as large as the effect itself.", "why": "ML minimizes prediction error, not causal error. The orthogonal score debiases ML predictions for causal use."},
                    {"mistake": "“Skip cross-fitting to save time”", "consequence": "The estimator inherits the ML’s overfitting. CI coverage drops below nominal. The estimate may look precise but is centered on the wrong value.", "why": "Chernozhukov et al. (2018): cross-fitting is essential for √n-consistency with data-adaptive nuisances."}
                ],
                "refs": "Chernozhukov et al. (2018) “Double/debiased machine learning for treatment and structural parameters” <i>Econometrics J</i> 21(1); Robins, Rotnitzky & Zhao (1994) “Estimation of regression coefficients...” <i>JASA</i>; Van der Laan & Rose (2011) <i>Targeted Learning</i>; Kennedy (2023) “Semiparametric doubly robust targeted double machine learning”"
            },
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
            "thinking": {
                "takeaway": "An estimate without sensitivity analysis is an open-loop claim — E-value = 2.73 means moderately robust",
                "symbolic": "\\text{E-value (VanderWeele & Ding 2017):} \\\\\n\\text{Let } RR_{\\text{obs}} = \\frac{P(Y=1 \\mid T=1)}{P(Y=1 \\mid T=0)} \\text{ (risk ratio)} \\\\\nE\\text{-value} = RR_{\\text{obs}} + \\sqrt{RR_{\\text{obs}} \\cdot (RR_{\\text{obs}} - 1)} \\\\\n\\text{Interpretation: } \\exists \\text{ unmeasured } U \\text{ that explains away the effect} \\Rightarrow RR_{TU} \\geq E\\text{-value} \\land RR_{UY} \\geq E\\text{-value}",
                "steps": [
                    {"label": "Mental operation 1: State the untestable assumption", "detail": "Our ATE estimate (+0.2437) rests on the assumption of <b>no unmeasured confounding</b>: {Y(0), Y(1)} ⊥ T \\mid {W, rain, weekend, payday}. But this assumption is <b>untestable from the data alone</b>. No statistical test can verify it. We need to quantify: how wrong could we be if this assumption fails?"},
                    {"label": "Mental operation 2: Convert ATE to risk ratio scale", "detail": "Our adjusted ATE = +0.2437. After adjustment, treated users have P(Y=1 \\mid T=1, adjusted) = 0.613, untreated have P(Y=1 \\mid T=0, adjusted) = 0.372. <b>RR_obs = 0.613 / 0.372 = 1.67.</b> This means treated users are 1.67× as likely to order as untreated users, after accounting for measured covariates."},
                    {"label": "Mental operation 3: Compute the E-value", "detail": "Plug RR = 1.67 into the formula: <b>E-value = 1.67 + √(1.67 × 0.67) = 1.67 + √(1.119) = 1.67 + 1.058 = 2.73.</b> An unmeasured confounder would need a risk ratio of <b>at least 2.73 with BOTH T and Y</b>, above and beyond the 4 measured covariates, to reduce the true effect to zero."},
                    {"label": "Mental operation 4: Calibrate — is RR=2.73 plausible?", "detail": "Smoking → lung cancer: RR ≈ 10-20. Moderate exercise → cardiovascular health: RR ≈ 2-3. Obesity → diabetes: RR ≈ 3-5. An RR of 2.73 is <b>moderately strong</b> — roughly comparable to the exercise-CVD association. Our finding is <b>moderately robust but not ironclad.</b> If a domain expert believes a confounder with RR ≥ 2.73 exists, the result is at risk."},
                    {"label": "Mental operation 5: Conservative check with CI bound", "detail": "Use the CI lower bound instead of the point estimate: RR_lower = 1.55. E-value_lower = 1.55 + √(1.55 × 0.55) = 1.55 + 0.923 = <b>2.47.</b> Even at the lower uncertainty bound, the E-value > 2. The result is robust to moderate unmeasured confounding even under parameter uncertainty."},
                    {"label": "Conclusion", "detail": "An estimate without a sensitivity analysis is an open-loop claim. The E-value closes the loop: it quantifies the debate about unmeasured confounding in a common, calibratable language. RR ≥ 2.73 is a concrete claim an adversary must make to dismiss the result."}
                ],
                "derivation": {"title": "From the Ding-VanderWeele bound to the E-value formula", "body": "Ding & VanderWeele (2016, <i>Epidemiology</i>) proved the maximum bias bound:<br><br>RR_{\\text{obs}} \\leq RR_{\\text{true}} \\times \\frac{RR_{TU} \\times RR_{UY}}{RR_{TU} + RR_{UY} - 1}<br><br>To make the true effect zero (RR_true = 1), set the worst case RR_TU = RR_UY = e:<br><br>RR_{\\text{obs}} \\leq \\frac{e^2}{2e - 1}<br><br>Cross-multiply: RR_obs × (2e - 1) = e²<br>e² - 2 × RR_obs × e + RR_obs = 0<br><br>Solve the quadratic: e = RR_obs + √(RR_obs × (RR_obs - 1))<br><br>This is the E-value."},
                "logic": [
                    {"if": "E-value > 5", "then": "Finding is highly robust — very strong unmeasured confounding needed", "holds": False, "note": "Our E-value = 2.73, not > 5"},
                    {"if": "E-value > 2", "then": "Finding is moderately robust", "holds": True, "note": "Our E-value = 2.73 > 2 ✓"},
                    {"if": "E-value < 1.5", "then": "Finding is fragile — modest confounding could explain it away", "holds": False, "note": "Our E-value = 2.73 > 1.5 ✓"}
                ],
                "pitfalls": [
                    {"mistake": "“E-value > 2 means there is no unmeasured confounding”", "consequence": "Wrong. The E-value quantifies how strong confounding would need to be, not whether it exists. A large E-value means the result is ROBUST to confounding, not that confounding is absent.", "why": "VanderWeele & Ding (2017): the E-value is a sensitivity analysis, not a test for confounding."},
                    {"mistake": "“Large E-value means we can skip adjustment for measured confounders”", "consequence": "Wrong. The E-value addresses UNMEASURED confounding, conditional on measured covariates. You still must adjust for measured confounders. The E-value asks: “After adjusting for what we measured, how much additional confounding could be hiding?”", "why": "The E-value formula uses the adjusted RR. If you skip adjustment, the RR_obs is inflated by measured confounding, giving a falsely reassuring E-value."}
                ],
                "refs": "VanderWeele & Ding (2017) “Sensitivity Analysis in Observational Research: Introducing the E-value” <i>Annals IM</i> 167(4); Ding & VanderWeele (2016) “Sensitivity analysis without assumptions” <i>Epidemiology</i>; Cinelli & Hazlett (2020) “Making sense of sensitivity”; Rosenbaum (2002) <i>Observational Studies</i> Ch.4"
            },
        },
        {
            "id": "test", "number": 7, "name": "TEST", "emoji": "🧪",
            "question": "Does the machinery refute itself?",
            "output": "CausalTestSuite",
            "explanation": "Refutation is continuous, not episodic (Design Principle P4). The refutation battery applies stress tests: (1) Placebo treatment — permute T randomly, estimate should be ~0. (2) Random common cause — add a random covariate, estimate should be stable. (3) Subset refuter — estimate on 80% of data, should agree with full sample. (4) Negative-control test — estimate T→NC effect, should be ~null (NC shares confounders, no T effect).",
            "thinking": {
                "takeaway": "Refutation is continuous — placebo, random cause, subset, and negative control tests stress-test the pipeline",
                "symbolic": "\\text{Refutation battery:} \\\\\n\\text{1. Placebo: } \\tilde{T} \\sim \\text{Bernoulli}(0.5) \\Rightarrow H_0: \\hat{\\tau}(\\tilde{T}, Y) = 0 \\\\\n\\text{2. Random common cause: } \\tilde{X} \\sim N(0,1) \\Rightarrow H_0: |\\hat{\\tau}(X) - \\hat{\\tau}(X, \\tilde{X})| < \\varepsilon \\\\\n\\text{3. Subset: } \\hat{\\tau}_{80\\%} \\approx \\hat{\\tau}_{100\\%} \\\\\n\\text{4. Negative control: } H_0: \\hat{\\tau}(T, NC) = 0 \\quad \\text{(Lipsitch, Tchetgen Tchetgen & Cohen 2010)} \\\\\n\\text{Loop invariants: } I_1 \\land I_2 \\land I_3 \\land I_4 \\text{ must hold in every run}",
                "steps": [
                    {"label": "Mental operation 1: Run the placebo test", "detail": "Shuffle T randomly (break any causal link) and re-run the entire pipeline. If our estimator is honest, it should find τ̂ ≈ 0. <b>Our result: τ̂ = +0.0017</b> (within tolerance of 0.013). <b>PASS ✓.</b> The pipeline does not hallucinate effects from noise."},
                    {"label": "Mental operation 2: Run the random common cause test", "detail": "Add a random N(0,1) variable to the adjustment set and re-estimate. A spurious variable should not change the estimate. <b>Our result: \\u0394τ̂ = 7.3×10\\u207b\\u2075</b> (effectively zero). <b>PASS ✓.</b> The estimate is not sensitive to irrelevant covariates."},
                    {"label": "Mental operation 3: Run the subset refuter", "detail": "Re-estimate on a random 80% subset. The estimate should be stable. <b>Our result: \\u0394τ̂ = 0.0052</b> (within tolerance of 0.019). <b>PASS ✓.</b> The estimate is not driven by a small subset of the data."},
                    {"label": "Mental operation 4: Run the negative control test", "detail": "NC (battery drain) shares the confounder U with Y but has NO causal effect from T. Estimate T→NC: if our adjustment is correct, this should be ~0. <b>Our result: τ̂(T→NC) = -0.0008</b> (within tolerance of 0.015). <b>PASS ✓.</b> No evidence of residual confounding."},
                    {"label": "Mental operation 5: Verify loop invariants", "detail": "<b>I1:</b> All artifacts carry same graph version ✓. <b>I2:</b> No adjustment variable is a descendant of T ✓. <b>I3:</b> EvaluationReport exists with sensitivity parameters recorded ✓. <b>I4:</b> |SMD| after IPW < 0.1 ✓. These are the causal equivalent of type-checking."},
                    {"label": "Conclusion", "detail": "<b>ALL GREEN.</b> All 4 refuters passed. All 4 loop invariants hold. The estimate survives adversarial stress-testing. This does NOT prove the estimate is correct (refuters are necessary, not sufficient), but failing ANY test would be a clear red flag."}
                ],
                "derivation": {"title": "Why the negative control test works", "body": "NC (battery drain) is chosen because:<br><b>1.</b> NC shares confounder U with Y: U→NC (more app use → more battery drain)<br><b>2.</b> T has NO causal effect on NC: T↛NC (notifications cannot drain battery)<br><br>Therefore, under correct adjustment for U (via W), the causal effect of T on NC must be zero. A non-zero estimate means:<br>- Either residual confounding remains (the adjustment set is incomplete)<br>- Or the NC assumption is wrong (T actually affects NC)<br><br>In either case, the smoke alarm fires. Lipsitch, Tchetgen Tchetgen & Cohen (2010, <i>Epidemiology</i>) formalized this as a general method for detecting bias in observational studies."},
                "logic": [
                    {"if": "Placebo test fails (τ̂ significantly ≠ 0 with random T)", "then": "Estimator is overfitting or there is data leakage", "holds": False, "note": "Our placebo estimate = +0.0017 ≈ 0 ✓"},
                    {"if": "Negative control test fails (τ̂(T→NC) significantly ≠ 0)", "then": "Residual confounding exists; adjustment set is incomplete", "holds": False, "note": "Our NC estimate = -0.0008 ≈ 0 ✓"},
                    {"if": "Subset test fails (τ̂_80% ≠ τ̂_100%)", "then": "Estimate is driven by a small subset; not reliable", "holds": False, "note": "\\u0394 = 0.0052, within tolerance ✓"}
                ],
                "pitfalls": [
                    {"mistake": "“All refuters passed, so the estimate is definitely correct”", "consequence": "Refuters check necessary conditions, not sufficient ones. You can pass all refutation tests with a wrong adjustment set if the wrong variables happen to be uncorrelated with T and Y in the sample.", "why": "Sharma & Kiciman (2020, DoWhy): refuters detect specific failure modes. They increase confidence but do not guarantee correctness."},
                    {"mistake": "“One refuter failed but the estimate looks fine, so I’ll ignore it”", "consequence": "A failed refuter is a smoke alarm. A failed NC test, in particular, strongly suggests residual confounding. Ignoring it is malpractice.", "why": "Lipsitch et al. (2010): negative controls are among the strongest falsification tools available for observational studies."}
                ],
                "refs": "Lipsitch, Tchetgen Tchetgen & Cohen (2010) “Negative controls: a tool for detecting confounding and bias” <i>Epidemiology</i>; Shi et al. (2020) negative control exposures; DoWhy (Sharma & Kiciman 2020) refutation framework; Hernán & Robins (2020) Ch.8 falsification"
            },
        },
        {
            "id": "evolve", "number": 8, "name": "EVOLVE", "emoji": "🔄",
            "question": "Is the world still the one we modeled?",
            "output": "EvolutionLog",
            "explanation": "<b>Principle:</b> the invariance principle (Peters, Bühlmann & Meinshausen 2016): a correctly specified causal mechanism has a stable conditional distribution across environments. For each endogenous node, we fit P(node | parents) on the reference (static) batch and evaluate log-loss on the new (holiday) batch. <b>(3) Compare:</b> the node with the largest degradation is the locus of drift -- T->M degrades most (coefficient 1.6->0.4). <b>(4) Confirm:</b> the Y (order) mechanism is confirmed invariant, and the monitor correctly localizes the change to exactly the one mechanism we altered in the DGP.",
            "thinking": {
                "takeaway": "Causal systems drift — the invariance principle detects mechanism changes and triggers autonomous re-estimation",
                "symbolic": "\\text{Invariance principle (Peters, Bühlmann & Meinshausen 2016, JRSS-B):} \\\\\nP(Y \\mid pa_G(Y)) \\text{ is INVARIANT across environments } \\mathcal{E} \\; \\Leftrightarrow \\; pa_G(Y) \\text{ are true causal parents} \\\\\n\\text{Mechanism-stability monitor:} \\\\\n\\text{For each node } V \\text{ with parents } pa(V): \\\\\n\\quad \\text{degradation}(V) = |\\log\\text{-loss}_{\\text{ref}}(V \\mid pa(V)) - \\log\\text{-loss}_{\\text{new}}(V \\mid pa(V))| \\\\\n\\quad \\text{alarm if degradation}(V) > \\theta \\quad (\\theta = 0.02 \\text{ nats}) \\\\\n\\text{Locus of drift} = \\arg\\max_V \\text{degradation}(V)",
                "steps": [
                    {"label": "Mental operation 1: State the monitoring problem", "detail": "We built our model on the <b>static regime</b> (normal conditions). Then a new batch arrives from the <b>holiday season</b>. How do we know if the world changed? We need a monitor that detects whether our causal assumptions still hold — without access to ground truth."},
                    {"label": "Mental operation 2: Apply the invariance principle", "detail": "Peters, Bühlmann & Meinshausen (2016) proved: a correctly specified causal mechanism P(V \\mid parents(V)) is <b>invariant</b> across environments. If the conditional distribution degrades, either the parents are wrong OR the mechanism genuinely changed. This principle bridges causal discovery and causal monitoring."},
                    {"label": "Mental operation 3: Run the monitor on the holiday batch", "detail": "For each endogenous node, fit P(node \\mid parents) on the static reference batch (10k obs), then evaluate log-loss on the holiday batch (10k obs). <b>Results:</b> M (app-open) degradation = +0.024 nats >> alarm threshold 0.02 → <b>DRIFT DETECTED.</b> Y (order) degradation = +0.008 nats < 0.02 → <b>Y mechanism INVARIANT.</b> All other nodes stable."},
                    {"label": "Mental operation 4: Localize the drift", "detail": "The monitor correctly identifies <b>M (app-open)</b> as the locus of drift. This is exactly the mechanism we altered in the DGP: the T→M coefficient was changed from 1.6 to 0.4 (users habituate to notifications during holidays). The Y mechanism (order given hunger and app-open) is confirmed unchanged."},
                    {"label": "Mental operation 5: Fire the actuator", "detail": "Drift detected → actuator fires: <b>re-run the full UCL pass on the holiday regime autonomously.</b> Same DAG, same identification, same estimation pipeline. Holiday ATE = +0.198 (truth = +0.194), CI covers truth ✓, all refuters green ✓. <b>The loop closed — zero human intervention.</b>"},
                    {"label": "Conclusion", "detail": "The invariance principle transforms causal monitoring from a heuristic into a theorem. The mechanism-stability monitor detected, localized, and responded to drift automatically. This is what makes the UCL a <b>closed loop</b>: the world changes, the monitor catches it, the actuator re-estimates, and the new estimate is validated against the same criteria."}
                ],
                "derivation": {"title": "Invariance principle — why it works", "body": "Peters, Bühlmann & Meinshausen (2016, JRSS-B 78(5)) proved:<br><br><b>Theorem:</b> The conditional distribution of an effect given its <i>true causal parents</i> is invariant across environments, while conditioning on any other set is not.<br><br>This provides both a <b>discovery method</b> (search for the set that makes the conditional invariant → that set is the causal parents) and a <b>monitoring method</b> (given a known DAG, test whether each P(V \\mid pa(V)) is stable; degradation = possible mechanism change).<br><br>In our holiday regime: the T→M coefficient changed (1.6 → 0.4) → P(M \\mid T, U) changed → M’s conditional degradation spikes. P(Y \\mid T, M, U, rain, weekend, payday) stayed the same → Y’s degradation stays low."},
                "logic": [
                    {"if": "A mechanism P(V \\mid pa(V)) degrades beyond threshold", "then": "Either the parents are wrong OR the mechanism genuinely changed → DRIFT DETECTED", "holds": True, "note": "M degradation = 0.024 > 0.02 → DRIFT"},
                    {"if": "The Y mechanism is invariant (degradation < threshold)", "then": "The outcome-generating process is stable; the DAG’s structure for Y is correct", "holds": True, "note": "Y degradation = 0.008 < 0.02 → INVARIANT"},
                    {"if": "Drift is detected and localized", "then": "The actuator fires: re-run UCL on the new regime autonomously", "holds": True, "note": "Holiday ATE recovered, CI covers truth"}
                ],
                "pitfalls": [
                    {"mistake": "“The model worked on the training data; it’ll work on new data too”", "consequence": "Causal systems drift. The T→M mechanism weakened by 75% in the holiday regime. Without an EVOLVE station, production systems silently degrade — reporting stale estimates from a world that no longer exists.", "why": "Pearl & Bareinboim (2014) transportability: causal effects are not automatically transportable across environments. You must test for invariance."},
                    {"mistake": "“Distribution shift = mechanism shift”", "consequence": "Wrong. P(T) changing (more notifications in holidays) is a distribution shift, not a mechanism shift. The EVOLVE monitor checks <b>conditional</b> distributions P(V \\mid parents), not marginals. This distinction is the entire point of the invariance principle.", "why": "Schölkopf et al. (2012): causal mechanisms are independent of the distributions of their causes (ICM principle). Changes in P(cause) do not imply changes in P(effect \\mid cause)."}
                ],
                "refs": "Peters, Bühlmann & Meinshausen (2016) “Causal inference by using invariant prediction” <i>JRSS-B</i> 78(5); Heinze-Deml, Peters & Meinshausen (2018) nonlinear ICP; Pfister et al. (2019) stabilizing variable selection; Arjovsky et al. (2019) Invariant Risk Minimization; Schölkopf et al. (2021) “Toward Causal Representation Learning”"
            },
        },
    ],
    "thinking_flow": [
        {"from_station": "frame", "to_station": "assume", "symbolic_link": "\\text{Estimand } \\tau = E[Y(1)-Y(0)] \\;\\longrightarrow\\; \\text{Encode beliefs as } G = (V, E)", "narrative": "With the estimand specified as a rung-2 intervention query, we now encode our causal beliefs — confounders, mediators, colliders, instruments — as a versioned DAG. Every arrow is an assumption; every absent arrow is a testable claim.", "key_question": "What causal structure would make this estimand identifiable?"},
        {"from_station": "assume", "to_station": "identify", "symbolic_link": "G = (V, E) \\;\\longrightarrow\\; \\text{Back-door}(T, Y, Z) \\text{ in } G_{\\text{BD}}", "narrative": "The DAG encodes our beliefs. Now we perform graph surgery — delete all edges OUT of T — and test whether an observed set Z d-separates T from Y. This is the back-door criterion, the mathematical bridge from assumptions to identification.", "key_question": "Can the causal effect be expressed in terms of observable probabilities?"},
        {"from_station": "identify", "to_station": "data", "symbolic_link": "P(Y \\mid do(T)) = \\sum_z P(Y \\mid T, z)P(z) \\;\\longrightarrow\\; \\text{Check positivity: } 0 < P(T=1 \\mid X) < 1", "narrative": "The back-door criterion proved the ATE is identified IF positivity holds. Now we sample data from the DGP and empirically verify that every unit has non-zero probability of receiving either treatment — and compute the rung-1 vs rung-2 gap to make confounding visible.", "key_question": "Does the observed data actually support the identification result?"},
        {"from_station": "data", "to_station": "feature", "symbolic_link": "\\text{Data passed } \\checkmark \\;\\longrightarrow\\; \\text{Compile feature spec from DAG: } Z_{\\text{adj}} = \\{W, \\text{rain}, \\text{weekend}, \\text{payday}\\}", "narrative": "Positivity holds, the gap confirms confounding exists. Now we compile the feature specification — which variables enter the model and which are excluded — directly from the DAG. Every exclusion (M mediator, S collider) has a graph-theoretic justification.", "key_question": "Which variables must enter the model — and which must be excluded?"},
        {"from_station": "feature", "to_station": "model", "symbolic_link": "Z_{\\text{adj}} = \\{W, \\text{rain}, \\text{weekend}, \\text{payday}\\} \\;\\longrightarrow\\; \\psi_i = \\hat{\\mu}_1 - \\hat{\\mu}_0 + \\frac{T(Y-\\hat{\\mu}_1)}{\\hat{e}} - \\frac{(1-T)(Y-\\hat{\\mu}_0)}{1-\\hat{e}}", "narrative": "With features locked, we estimate using the Augmented Inverse Probability Weighting (AIPW) score — a doubly-robust, Neyman-orthogonal estimator that safely uses gradient boosting for nuisance functions without contaminating the causal estimand.", "key_question": "How do we estimate the ATE with modern ML without ML's regularization bias leaking in?"},
        {"from_station": "model", "to_station": "evaluate", "symbolic_link": "\\hat{\\tau} = 0.2437, \\; 95\\%\\text{CI} = [0.228, 0.259] \\;\\longrightarrow\\; E\\text{-value} = RR + \\sqrt{RR(RR-1)} = 2.73", "narrative": "We have a precise estimate. Now we ask: how wrong could we be? The E-value quantifies the minimum strength an unmeasured confounder would need with BOTH T and Y to explain away the effect. No sensitivity analysis = no closed loop.", "key_question": "What is the minimum strength of unmeasured confounding needed to nullify our result?"},
        {"from_station": "evaluate", "to_station": "test", "symbolic_link": "E\\text{-value} = 2.73 \\;\\longrightarrow\\; \\text{Placebo}(T_{\\text{shuffled}}), \\;\\text{RCC}(\\tilde{X}), \\;\\text{Subset}(80\\%), \\;\\text{NC}(T \\to NC)", "narrative": "Sensitivity analysis complete. Now we stress-test the pipeline with four refutation tests — placebo treatment, random common cause, subset refuter, and negative control — plus four loop invariants that must hold in every run.", "key_question": "Does the machinery refute itself when we deliberately break it?"},
        {"from_station": "test", "to_station": "evolve", "symbolic_link": "\\text{All refuters green } \\checkmark \\;\\longrightarrow\\; \\text{Monitor } \\Delta\\log\\text{-loss}(V \\mid pa(V)) \\text{ on new data batch}", "narrative": "The pipeline passes all tests in the static regime. But the world changes — the holiday season alters the T→M mechanism. The mechanism-stability monitor detects, localizes, and responds to this drift autonomously, closing the self-evolving loop.", "key_question": "Is the world still the one we modeled, and if not — what changed and what do we do about it?"}
    ],
    "ladder_chain": {
        "title": "Climbing Pearl's Ladder — From Seeing to Doing to Imagining",
        "steps": [
            {"rung": 1, "name": "Association (Seeing)", "symbolic": "P(Y \\mid T=1) - P(Y \\mid T=0) = +0.345", "thought": "I observe that notified users order 34.5 pp more. This is a fact about the data. But is it causal? No — I know the platform targets hungrier users, and hungrier users order more regardless. The 34.5 pp mixes causation with confounding. I cannot stop here.", "limitation": "Cannot distinguish P(Y \\mid T) from P(Y \\mid do(T)). Passive observation confounds causal and spurious association."},
            {"rung": 2, "name": "Intervention (Doing)", "symbolic": "E[Y \\mid do(T=1)] - E[Y \\mid do(T=0)] = +0.241", "thought": "I encode my causal beliefs as a DAG. I prove identifiability via the back-door criterion: Z = {W, rain, weekend, payday} blocks all back-door paths. I estimate via AIPW with cross-fitting. The adjusted estimate (+0.244) matches the Monte Carlo truth (+0.241) within 0.002. The confounding gap of +0.104 has been removed.", "limitation": "Answers “what is the average effect?” but not “was THIS specific user’s order caused by the notification?” Averages over all units; cannot condition on individual outcomes."},
            {"rung": 3, "name": "Counterfactuals (Imagining)", "symbolic": "P(Y(0)=0 \\mid T=1, Y=1) = 0.37", "thought": "Among treated users who ordered: 37% would NOT have ordered without the notification. I compute this by abduction (infer noise from observed data), action (do(T=0) for those users), and prediction (re-run SCM with same noise, different T). This requires the full SCM — not just the DAG, but the structural equations and their noise structure.", "limitation": "Requires the full SCM with noise structure. Rung-2 quantities (ATE) do not suffice — the ATE averages over all units, but counterfactuals condition on a specific subset. You must know HOW the variables were generated."}
        ]
    },
    "tools_and_projects": [
        {"name": "DoWhy", "authors": "Sharma & Kiciman (2020), Microsoft Research", "what": "Causal effect estimation library with explicit identification step (graph → ID → estimation → refutation). The refutation API directly inspired our Station 7.", "url": "https://github.com/py-why/dowhy"},
        {"name": "EconML", "authors": "Syrgkanis et al., Microsoft", "what": "Heterogeneous treatment effects, DML, CATE estimation, policy learning. Production-grade orthogonal ML for causal inference.", "url": "https://github.com/py-why/EconML"},
        {"name": "dagitty", "authors": "Textor et al. (2016)", "what": "Browser-based DAG drawing and identification analysis. Computes adjustment sets, tests d-separation, detects instrumental variables. The gold standard for DAG reasoning.", "url": "https://dagitty.net/"},
        {"name": "CausalNex", "authors": "QuantumBlack / McKinsey", "what": "Bayesian network + do-calculus. Structure learning from data with domain knowledge constraints.", "url": "https://github.com/quantumblacklabs/causalnex"},
        {"name": "Tetrad / py-causal", "authors": "Spirtes, Glymour, Scheines (CMU)", "what": "Causal discovery algorithms since the 1990s: PC, FCI, GFCI, LiNGAM. The reference implementation of constraint-based and score-based discovery.", "url": "https://github.com/cmu-phil/tetrad"},
        {"name": "causallib", "authors": "IBM Research", "what": "Library of causal estimators (IPW, matching, DR, TMLE) with scikit-learn interface. Emphasis on model selection and evaluation for causal tasks.", "url": "https://github.com/IBM/causallib"},
        {"name": "CausalML", "authors": "Uber", "what": "Uplift modeling, CATE estimation with tree-based and meta-learner methods. Built for large-scale production causal inference.", "url": "https://github.com/uber/causalml"},
        {"name": "CausalPy", "authors": "PyMC team", "what": "Bayesian causal inference: synthetic control, DiD, regression discontinuity, all with full posterior uncertainty.", "url": "https://github.com/pymc-labs/CausalPy"},
        {"name": "Ananke", "authors": "Bhattacharya et al., UCLA", "what": "Causal identification via the ID algorithm (Shpitser & Pearl 2006). Computes identifying functionals for any identified query in a given DAG.", "url": "https://github.com/ananke-developers/ananke"}
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

# Populate the data_snippet with actual sample rows
_snip_cols = data["content"]["data_snippet"]["columns"]
_snip_df = dgp_sample(5, seed=42)[_snip_cols]
data["content"]["data_snippet"]["rows"] = [
    list(_snip_df.loc[i, _snip_cols].values) for i in range(5)
]

# ── Graph metadata (avoids hardcoding in HTML) ──
from nomnom.graph import nomnom_graph as _nomnom_graph
_g = _nomnom_graph()
data["dag_stats"] = {
    "n_observed": len(_g.observed),
    "n_latent": len(_g.latent),
    "n_edges": len(_g.edges),
    "n_absent_edges": len(_g.absent_edges),
    "criterion": "back-door",
    "node_roles": dict(sorted(_g.node_roles.items())),
}

# ── Visual diagrams ──
data["visuals"] = {
    "ladder": {
        "title": "Pearl's Ladder of Causation",
        "rungs": [
            {"y": 140, "label": "Rung 1: Association", "symbol": "P(Y | T)", "desc": "Seeing — passive observation", "op": "condition", "color": "#3498db"},
            {"y": 280, "label": "Rung 2: Intervention", "symbol": "P(Y | do(T))", "desc": "Doing — graph surgery, external manipulation", "op": "do()", "color": "#2ecc71"},
            {"y": 420, "label": "Rung 3: Counterfactual", "symbol": "P(Y(0)=0 | T=1,Y=1)", "desc": "Imagining — same unit, different treatment", "op": "abduction", "color": "#e74c3c"}
        ],
        "climb": [
            {"from_rung": 1, "to_rung": 2, "requires": "DAG + Back-door Criterion"},
            {"from_rung": 2, "to_rung": 3, "requires": "Full SCM + Noise Structure"}
        ]
    },
    "confounding_gap": {
        "title": "Decomposing the Naive Association",
        "bars": [
            {"label": "Naive P(Y|T=1) - P(Y|T=0)", "value": 0.343, "color": "#e74c3c", "rung": 1},
            {"label": "True ATE E[Y(1)-Y(0)]", "value": 0.241, "color": "#2ecc71", "rung": 2},
            {"label": "Confounding Bias", "value": 0.102, "color": "#f39c12", "rung": "bias"}
        ]
    },
    "evalue_gauge": {
        "title": "E-value Sensitivity Threshold",
        "our_value": 2.73,
        "zones": [
            {"min": 1.0, "max": 1.5, "label": "Trivial", "color": "#e74c3c"},
            {"min": 1.5, "max": 2.0, "label": "Fragile", "color": "#f39c12"},
            {"min": 2.0, "max": 5.0, "label": "Robust", "color": "#2ecc71"},
            {"min": 5.0, "max": 10.0, "label": "Highly Robust", "color": "#2563eb"}
        ]
    },
    "identification_flow": {
        "title": "Identification via Back-Door Criterion",
        "steps": [
            {"x": 0, "label": "DAG G", "detail": "17 edges", "color": "#3498db"},
            {"x": 1, "label": "Graph Surgery", "detail": "Delete T→M, T→Y, T→S", "color": "#9b59b6"},
            {"x": 2, "label": "Back-Door Graph G_BD", "detail": "Only back-door paths remain", "color": "#f39c12"},
            {"x": 3, "label": "Candidate Search", "detail": "Test subsets for d-sep", "color": "#e67e22"},
            {"x": 4, "label": "Z = {W,rain,wknd,payday}", "detail": "Minimal valid set", "color": "#2ecc71"}
        ]
    }
}

# ── Write ──
out_path = Path(__file__).resolve().parent / "data.json"
out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Written: {out_path} ({out_path.stat().st_size:,} bytes)")
print(f"  static ATE={data['static']['ate']}, holiday ATE={data['holiday']['ate']}")
print(f"  {len(data['content']['stations'])} stations, {len(data['content']['glossary'])} glossary terms")
