"""Populate the Causal Science Knowledge Graph from existing project data.

Sources:
  1. data.json — glossary, stations, DAG structure
  2. nomnom/graph.py — variable roles, edges, absent edges
  3. notebooks/math_bridge/ — 6 levels with bridge insights
  4. notebooks/tier1_gallery/ — 6 gallery cases
  5. ref/literature_review.md — references (future: parse bibliography)
  6. ucl/contracts/artifacts.py — station contract types

Usage:
  python kg/populate.py          # full population
  python kg/populate.py --dry    # dry run (print Cypher, don't execute)
"""
import json, sys, os
from pathlib import Path
from textwrap import dedent

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "causal123")

# ── Load project data ──
demo = json.loads((REPO / "docs" / "demo" / "data.json").read_text(encoding="utf-8"))
content = demo["content"]
from nomnom.graph import nomnom_graph
g = nomnom_graph()

# =============================================================================
# Helper
# =============================================================================
class KG:
    def __init__(self, uri=URI, auth=AUTH, dry=False):
        self.dry = dry
        if not dry:
            self.driver = GraphDatabase.driver(uri, auth=auth)
            self.driver.verify_connectivity()
            print("Connected to Neo4j")

    def run(self, cypher, **params):
        if self.dry:
            print(f"  CYPHER: {cypher[:120]}...")
            return
        with self.driver.session() as s:
            s.run(cypher, **params)

    def clear(self):
        self.run("MATCH (n) DETACH DELETE n")
        print("Cleared all nodes")

    def close(self):
        if not self.dry and hasattr(self, "driver"):
            self.driver.close()


# =============================================================================
# Population functions
# =============================================================================
def populate_concepts(kg):
    """Extract from demo glossary + stations + deep dives."""
    concepts = [
        # Core estimands
        {"id":"ate","name":"Average Treatment Effect (ATE)","rung":2,
         "definition":"E[Y(1) − Y(0)]: the expected difference in outcomes if everyone vs. no one received treatment.",
         "formal_def":"\\tau = E[Y(1) - Y(0)]","glossary":True},
        {"id":"cate","name":"Conditional ATE (CATE)","rung":2,
         "definition":"ATE conditioned on covariates: E[Y(1)−Y(0) | X=x]. Captures treatment effect heterogeneity.",
         "formal_def":"\\tau(x) = E[Y(1)-Y(0) \\mid X=x]","glossary":False},
        {"id":"late","name":"Local ATE (LATE)","rung":2,
         "definition":"ATE for compliers in an IV design. Identified when the instrument satisfies relevance, exclusion, and independence.",
         "formal_def":"LATE = E[Y(1)-Y(0) \\mid complier]","glossary":False},
        # Identification
        {"id":"back-door-criterion","name":"Back-Door Criterion","rung":2,
         "definition":"A set Z satisfies the back-door criterion relative to (T,Y) if no node in Z is a descendant of T, and Z d-separates T from Y in the back-door graph (Pearl 1995, Def. 3.3.1).",
         "formal_def":"T \\perp\\!\\!\\!\\perp_{G_{BD}} Y \\mid Z","glossary":True,
         "aka":["back-door","backdoor criterion"]},
        {"id":"front-door-criterion","name":"Front-Door Criterion","rung":2,
         "definition":"Identifies the ATE when a mediator M intercepts all causal paths from T to Y and shares no confounders with Y. More fragile than back-door (requires no T→Y direct edge).",
         "formal_def":"T \\rightarrow M \\rightarrow Y, T \\perp Y \\mid do(M)","glossary":False},
        {"id":"do-calculus","name":"do-Calculus","rung":2,
         "definition":"Three rules (R1–R3) that transform expressions involving the do-operator into observational quantities. Complete for identifying any identifiable causal effect (Shpitser & Pearl 2006).",
         "formal_def":"R1: P(y\\mid do(x),z,w)=P(y\\mid do(x),w)\\text{ if }Y\\perp Z\\mid X,W\\text{ in }G_{\\bar{X}}","glossary":False},
        # Core concepts
        {"id":"d-separation","name":"d-Separation","rung":1,
         "definition":"Graphical test for conditional independence: a path is blocked if it contains a chain/fork with the middle node conditioned on, or a collider with neither the collider nor any descendant conditioned on.",
         "formal_def":"X \\perp\\!\\!\\!\\perp_G Y \\mid Z","glossary":True,
         "aka":["d-sep","d-separated"]},
        {"id":"confounding","name":"Confounding","rung":2,
         "definition":"A common cause of T and Y that creates spurious association. Without adjustment for confounders, P(Y|T) ≠ P(Y|do(T)).",
         "formal_def":"\\exists U: U\\rightarrow T, U\\rightarrow Y","glossary":True},
        {"id":"collider-bias","name":"Collider Bias (Berkson's Paradox)","rung":2,
         "definition":"Conditioning on a common effect of T and Y opens a spurious path, creating association where none exists. Never condition on a collider or its descendants.",
         "formal_def":"T\\rightarrow S\\leftarrow Y, \\text{ condition on }S\\Rightarrow T\\not\\perp Y","glossary":True},
        {"id":"mediation","name":"Mediation","rung":2,
         "definition":"A variable M on the causal path from T to Y (T→M→Y). Adjusting for M blocks the indirect effect — only appropriate when estimating the direct effect.",
         "formal_def":"T\\rightarrow M\\rightarrow Y","glossary":True},
        {"id":"instrumental-variable","name":"Instrumental Variable (IV)","rung":2,
         "definition":"Z affects T (relevance), has no direct effect on Y except through T (exclusion), and is independent of confounders. Identifies LATE for compliers.",
         "formal_def":"Z\\rightarrow T, Z\\not\\rightarrow Y, Z\\perp U","glossary":True},
        {"id":"negative-control","name":"Negative Control","rung":2,
         "definition":"A variable that shares the same confounders as the outcome but has no causal effect from treatment. A non-zero estimated T→NC effect signals residual confounding (Lipsitch et al. 2010).",
         "formal_def":"U\\rightarrow NC, T\\not\\rightarrow NC","glossary":True},
        {"id":"sensitivity-analysis","name":"Sensitivity Analysis","rung":2,
         "definition":"Quantifies how strong unmeasured confounding would need to be to explain away an observed effect. The E-value (VanderWeele & Ding 2017) is the most widely used metric.",
         "glossary":False},
        {"id":"positivity","name":"Positivity (Overlap)","rung":2,
         "definition":"Every unit must have non-zero probability of receiving either treatment: 0 < P(T=1|X=x) < 1 for all x. Without it, ATE requires extrapolation.",
         "formal_def":"0 < P(T=1\\mid X=x) < 1\\;\\forall x","glossary":True},
        {"id":"ignorability","name":"Ignorability (Unconfoundedness)","rung":2,
         "definition":"{Y(0),Y(1)} independent of T given X: all common causes of T and Y are measured and adjusted for. Fundamentally untestable from observational data.",
         "formal_def":"\\{Y(0),Y(1)\\} \\perp T \\mid X","glossary":True,
         "aka":["unconfoundedness","conditional exchangeability","no unmeasured confounding"]},
        {"id":"sutva","name":"SUTVA","rung":2,
         "definition":"Stable Unit Treatment Value Assumption: (1) no interference between units, (2) consistency — the observed Y equals the potential outcome under the treatment actually received.",
         "formal_def":"Y_i = T_i Y_i(1) + (1-T_i)Y_i(0)","glossary":True},
        {"id":"neyman-orthogonality","name":"Neyman Orthogonality","rung":2,
         "definition":"The derivative of the expected score with respect to nuisance parameters is zero at the true values. First-order errors in nuisance models cancel out rather than contaminating the causal estimand (Chernozhukov et al. 2018).",
         "formal_def":"\\partial/\\partial\\eta E[\\psi(T,Y,X;\\theta,\\eta)]|_{\\eta=\\eta_0}=0","glossary":False},
        {"id":"double-robustness","name":"Double Robustness","rung":2,
         "definition":"An estimator is doubly robust if it is consistent when EITHER the propensity model OR the outcome model is correctly specified — not both. AIPW and TMLE are doubly robust.",
         "glossary":False},
        {"id":"invariance-principle","name":"Invariance Principle","rung":2,
         "definition":"A correctly specified causal mechanism P(effect | direct causes) is invariant across environments (Peters, Buhlmann & Meinshausen 2016). Used for causal discovery and drift detection.",
         "glossary":False},
        {"id":"simpsons-paradox","name":"Simpson's Paradox","rung":1,
         "definition":"An association that appears in aggregated data reverses or disappears within subgroups. The data alone cannot tell you whether to condition — that decision requires causal knowledge (which variable is the confounder?).",
         "glossary":False},
        {"id":"rung-1","name":"Rung 1 — Association","rung":1,
         "definition":"Pearl's first rung: seeing. P(Y|T) — passive observation. Conditional probabilities, correlation, regression. Cannot distinguish causation from confounding.",
         "formal_def":"P(Y\\mid T)","glossary":False},
        {"id":"rung-2","name":"Rung 2 — Intervention","rung":2,
         "definition":"Pearl's second rung: doing. P(Y|do(T)) — external manipulation, graph surgery. Answers 'what happens if we force T?' Requires causal assumptions (DAG).",
         "formal_def":"P(Y\\mid do(T))","glossary":False},
        {"id":"rung-3","name":"Rung 3 — Counterfactuals","rung":3,
         "definition":"Pearl's third rung: imagining. P(Y(0)=0 | T=1, Y=1) — same unit, different treatment. Requires the full SCM with noise structure. Answers 'was it the treatment that caused THIS outcome?'",
         "formal_def":"P(Y(t') \\mid T=t, Y=y)","glossary":False},
        # ── Statistics & Data Science foundations ──
        {"id":"law-of-total-probability","name":"Law of Total Probability","rung":1,
         "definition":"P(A) = sum_i P(A | B_i) P(B_i) for a partition {B_i}. The foundation of adjustment formulas, stratification, and g-computation.",
         "formal_def":"P(A) = \\sum_i P(A \\mid B_i) P(B_i)","glossary":False},
        {"id":"bayes-rule","name":"Bayes' Rule","rung":1,
         "definition":"P(H|D) = P(D|H)P(H)/P(D). Updates prior beliefs with evidence. The engine of Bayesian inference, posterior computation, and abduction (rung 3 counterfactuals).",
         "formal_def":"P(H\\mid D) = \\frac{P(D\\mid H)P(H)}{P(D)}","glossary":False},
        {"id":"central-limit-theorem","name":"Central Limit Theorem","rung":1,
         "definition":"The sampling distribution of the sample mean approaches N(mu, sigma^2/n) as n increases, regardless of the population distribution (under finite variance). Justifies normal-approximation CIs.",
         "formal_def":"\\sqrt{n}(\\bar{X}_n - \\mu) \\xrightarrow{d} N(0,\\sigma^2)","glossary":False},
        {"id":"maximum-likelihood","name":"Maximum Likelihood Estimation","rung":1,
         "definition":"Choose parameters that maximize the probability of observing the data. MLE is consistent, asymptotically normal, and efficient under regularity conditions. Foundation of most statistical modeling.",
         "formal_def":"\\hat{\\theta}_{MLE} = \\arg\\max_\\theta \\prod_i P(X_i \\mid \\theta)","glossary":False},
        {"id":"bias-variance-tradeoff","name":"Bias-Variance Tradeoff","rung":1,
         "definition":"MSE = Bias^2 + Variance + irreducible error. Regularization reduces variance at the cost of increased bias. In causal inference, regularization bias in nuisance models can contaminate the causal estimand (the nD trap).",
         "formal_def":"MSE(\\hat{\\theta}) = E[(\\hat{\\theta}-\\theta)^2] = Bias(\\hat{\\theta})^2 + Var(\\hat{\\theta})","glossary":False},
        {"id":"regularization","name":"Regularization","rung":1,
         "definition":"Adding a penalty term to the loss function (L1/L2/elastic net) to constrain model complexity. Prevents overfitting but introduces bias. In causal ML, this bias must be removed by orthogonal scores (DML) or targeting (TMLE).",
         "formal_def":"\\hat{\\beta} = \\arg\\min_\\beta \\{ -\\ell(\\beta) + \\lambda R(\\beta) \\}","glossary":False},
        {"id":"cross-validation","name":"Cross-Validation","rung":1,
         "definition":"Splitting data into folds for honest model evaluation. In causal inference, cross-fitting (Chernozhukov et al. 2018) uses held-out folds to prevent overfitting bias from contaminating causal estimates.",
         "formal_def":"CV_k = \\frac{1}{k}\\sum_{i=1}^k \\text{Loss}(\\hat{f}_{-i}, \\text{fold}_i)","glossary":False},
        {"id":"confidence-interval","name":"Confidence Interval","rung":1,
         "definition":"An interval [L, U] that covers the true parameter with probability 1-alpha over repeated sampling. In causal inference, the CI quantifies sampling uncertainty but NOT model/graph uncertainty.",
         "formal_def":"P(\\theta \\in [L,U]) = 1-\\alpha","glossary":False},
        {"id":"p-value","name":"P-value","rung":1,
         "definition":"The probability of observing a test statistic at least as extreme as the one computed, under the null hypothesis. In causal inference, refutation tests use p-values from placebo/randomization distributions, not from regression tables.",
         "formal_def":"p = P(T \\geq t_{obs} \\mid H_0)","glossary":False},
        {"id":"type-i-error","name":"Type I Error (False Positive)","rung":1,
         "definition":"Rejecting a true null hypothesis. In causal refutation, a placebo test that falsely rejects the null (tau=0) is a type I error — the pipeline claims an effect where none exists.",
         "formal_def":"\\alpha = P(\\text{reject } H_0 \\mid H_0 \\text{ true})","glossary":False},
        {"id":"statistical-power","name":"Statistical Power","rung":1,
         "definition":"Probability of correctly rejecting a false null. In causal inference, power calculations determine minimum detectable effect sizes and sample size requirements for planned studies.",
         "formal_def":"1-\\beta = P(\\text{reject } H_0 \\mid H_1 \\text{ true})","glossary":False},
        {"id":"frisch-waugh-lovell","name":"Frisch-Waugh-Lovell Theorem","rung":1,
         "definition":"The coefficient on X in a regression of Y on X and Z equals the coefficient from regressing Y (residualized on Z) on X (residualized on Z). Shows that regression 'partials out' controls — the math behind linear adjustment.",
         "formal_def":"\\hat{\\beta}_X = (X^\\top M_Z X)^{-1} X^\\top M_Z Y, \\; M_Z = I - Z(Z^\\top Z)^{-1}Z^\\top","glossary":False},
        {"id":"markov-condition","name":"Markov Condition","rung":1,
         "definition":"A node is independent of its non-descendants given its parents. The bridge between graph structure and probability: the joint distribution factorizes according to the DAG.",
         "formal_def":"X \\perp NonDescendants(X) \\mid pa(X) \\Rightarrow P(V) = \\prod_j P(V_j \\mid pa(V_j))","glossary":False},
        {"id":"exchangeability","name":"Exchangeability","rung":2,
         "definition":"The treated and untreated groups are comparable in their potential outcomes. In randomized experiments, exchangeability holds by design. In observational studies, it requires the ignorability assumption.",
         "formal_def":"\\{Y(0),Y(1)\\} \\perp T","glossary":False},
        {"id":"consistency-assumption","name":"Consistency Assumption","rung":2,
         "definition":"Y = T*Y(1) + (1-T)*Y(0): the observed outcome equals the potential outcome under the treatment actually received. Links counterfactual notation to observed data. Part of SUTVA.",
         "formal_def":"Y = T\\cdot Y(1) + (1-T)\\cdot Y(0)","glossary":False},
        # ── Cross-disciplinary foundations ──
        # Philosophy
        {"id":"humean-causation","name":"Humean Causation","rung":1,"glossary":False,
         "definition":"Hume (1748): causation is constant conjunction — we observe regular succession, not necessary connection. The original articulation of the problem that causal inference tries to solve. 'We never observe causes — only regularities.'"},
        {"id":"counterfactual-theory","name":"Counterfactual Theory of Causation","rung":3,"glossary":False,
         "definition":"Lewis (1973): 'X causes Y' means 'Y would not have occurred if X had not occurred' in the closest possible world. The philosophical foundation of the potential outcomes framework.",
         "formal_def":"X \\text{ causes } Y \\iff Y(\\neg X) \\neq Y(X) \\text{ in the closest world}"},
        {"id":"manipulability-theory","name":"Manipulability Theory","rung":2,"glossary":False,
         "definition":"Woodward (2003): X causes Y if intervening on X (while holding everything else fixed) changes Y. The philosophical foundation of Pearl's do-operator and the interventionist account of causation."},
        {"id":"cartwrights-dictum","name":"Cartwright's Dictum","rung":2,"glossary":False,
         "definition":"Cartwright (1999): 'No causes in, no causes out.' A causal model is only as good as the causal knowledge that went into it. Data alone, without causal assumptions, cannot produce causal conclusions."},
        # Econometrics
        {"id":"heckman-selection","name":"Heckman Selection Bias","rung":2,"glossary":False,
         "definition":"Bias from non-random selection into treatment: treated units differ systematically from untreated units in ways that affect outcomes. The econometric formulation of confounding (Heckman 1979).",
         "formal_def":"E[Y\\mid T=1] - E[Y\\mid T=0] = ATE + \\underbrace{E[Y(0)\\mid T=1] - E[Y(0)\\mid T=0]}_{\\text{selection bias}}"},
        {"id":"granger-causality","name":"Granger Causality","rung":1,"glossary":False,
         "definition":"X 'Granger-causes' Y if past values of X improve prediction of Y beyond past values of Y alone. Important: Granger causality is predictive, not causal — it does not imply X causes Y in the interventionist sense."},
        {"id":"panel-data-methods","name":"Panel Data Methods","rung":2,"glossary":False,
         "definition":"Fixed effects, random effects, and difference-in-differences exploit panel structure to control for unobserved time-invariant confounders. DiD is the workhorse causal method in econometrics."},
        # Epidemiology
        {"id":"bradford-hill-criteria","name":"Bradford Hill Criteria","rung":2,"glossary":False,
         "definition":"Nine criteria for assessing causality from observational evidence (Hill 1965): strength, consistency, specificity, temporality, biological gradient, plausibility, coherence, experiment, analogy. Pre-cursor to modern causal formalization but still widely taught."},
        {"id":"target-trial-emulation","name":"Target Trial Emulation","rung":2,"glossary":False,
         "definition":"Hernan & Robins (2016): specify the idealized RCT you would run, then emulate it from observational data. Prevents method-driven analysis. The foundation of modern causal epidemiology."},
        {"id":"confounding-in-epidemiology","name":"Confounding in Epidemiology","rung":2,"glossary":False,
         "definition":"A confounder must be (1) associated with exposure, (2) associated with outcome, (3) not on the causal path. Greenland, Pearl & Robins (1999) unified the epidemiological definition with the graphical one."},
        # Computer Science / AI
        {"id":"causal-representation-learning","name":"Causal Representation Learning","rung":2,"glossary":False,
         "definition":"Learning representations that capture causal structure rather than mere statistical associations. Scholkopf et al. (2021): the next frontier for robust, generalizable, fair ML systems."},
        {"id":"causal-reinforcement-learning","name":"Causal Reinforcement Learning","rung":2,"glossary":False,
         "definition":"RL agents that learn causal models of their environment to plan interventions and generalize across tasks. Combines Pearl's causal hierarchy with Sutton's RL framework. Active research area."},
        {"id":"fairness-in-ml","name":"Fairness in Machine Learning","rung":2,"glossary":False,
         "definition":"Causal definitions of fairness (counterfactual fairness, no unresolved discrimination) require causal models. Associational fairness criteria can be mutually inconsistent and fail to address root causes of bias."},
        # Psychology / Social Science
        {"id":"campbell-validity","name":"Campbell's Validity Framework","rung":2,"glossary":False,
         "definition":"Campbell & Stanley (1963): internal validity (is the effect causal?), external validity (does it generalize?), construct validity (are we measuring what we think?), statistical conclusion validity. The social-science precursor to modern causal design."},
        {"id":"mediation-in-psychology","name":"Mediation in Psychology","rung":2,"glossary":False,
         "definition":"Baron & Kenny (1986): the classic regression-based approach to mediation analysis. Modern causal mediation (VanderWeele, Imai, Pearl) generalizes this to allow interactions, nonlinearities, and identification from DAGs rather than regression coefficients."},
        {"id":"quasi-experiments","name":"Quasi-Experiments","rung":2,"glossary":False,
         "definition":"Research designs that approximate randomized experiments when randomization is infeasible: regression discontinuity, interrupted time series, difference-in-differences, instrumental variables. Shadish, Cook & Campbell (2002) is the canonical reference."},
        # Biostatistics
        {"id":"g-methods","name":"G-Methods","rung":2,"glossary":False,
         "definition":"Robins' generalized methods for time-varying confounding: g-computation (parametric), IPTW (weighting), g-estimation (structural nested models). The biostatistics foundation of modern longitudinal causal inference."},
        {"id":"propensity-score","name":"Propensity Score","rung":2,"glossary":False,
         "definition":"Rosenbaum & Rubin (1983): e(X) = P(T=1|X). The propensity score is a balancing score — conditioning on e(X) is sufficient to remove confounding. Reduces high-dimensional X to a scalar, enabling matching, stratification, and IPW."},
        {"id":"sensitivity-analysis-epi","name":"Sensitivity Analysis in Epidemiology","rung":2,"glossary":False,
         "definition":"Rosenbaum (2002): bounding the treatment effect under hypothetical unmeasured confounding. The E-value (VanderWeele & Ding 2017) is the modern, standardized version used throughout this project."},
        # Bayesian-causal bridge concepts
        {"id":"bayesian-marginalization","name":"Bayesian Marginalization","rung":1,"glossary":False,
         "definition":"P(prediction | data) = integral of P(prediction | theta) * P(theta | data) d-theta. The Bayesian engine for integrating out nuisance parameters. Structurally identical to causal adjustment (Sigma) but operates on P(Z|T) rather than P(Z) — conditions instead of intervening.",
         "formal_def":"P(\\tilde{y} \\mid D) = \\int P(\\tilde{y} \\mid \\theta) P(\\theta \\mid D) d\\theta"},
        {"id":"conditioning-vs-intervening","name":"Conditioning vs. Intervening","rung":2,"glossary":False,
         "definition":"Conditioning (Bayes): P(Y | T) = sum_z P(Y | T, z) P(z | T). Intervening (causal): P(Y | do(T)) = sum_z P(Y | T, z) P(z). Same Sigma, different weight. The difference between Rung 1 and Rung 2 is entirely in whether you use P(z|T) or P(z).",
         "formal_def":"\\text{Condition: } \\sum_z P(Y|T,z)P(z|T) \\quad \\text{vs.} \\quad \\text{Intervene: } \\sum_z P(Y|T,z)P(z)"},
        {"id":"truncated-factorization","name":"Truncated Factorization","rung":2,"glossary":False,
         "definition":"Pearl's formal definition of intervention: delete P(T|pa(T)) from the Markov factorization, fix T=t. P(V|do(T=t)) = prod_{Vj≠T} P(Vj | pa(Vj)) |_{T=t}. The general mechanism from which all identification formulas (back-door, front-door, IV) are derived by marginalizing over subsets of variables.",
         "formal_def":"P(V \\setminus \\{T\\} \\mid do(T=t)) = \\prod_{V_j \\neq T} P(V_j \\mid pa(V_j)) \\big|_{T=t}"},
        {"id":"id-algorithm","name":"ID Algorithm (Shpitser & Pearl 2006)","rung":2,"glossary":False,
         "definition":"A complete polynomial-time algorithm that determines whether any causal effect is identifiable from a given DAG and, if so, expresses it as a sequence of Sigma (marginalization) operations over observed variables. Every identifiable causal effect is a Sigma expression."},
    ]
    for c in concepts:
        kg.run("""
            MERGE (n:Concept {id: $id})
            SET n.name = $name, n.definition = $definition, n.rung = $rung
            SET n.glossary = $glossary, n.formal_def = $formal_def, n.aka = $aka
        """, id=c["id"], name=c["name"], definition=c["definition"], rung=c["rung"],
             glossary=c.get("glossary", False), formal_def=c.get("formal_def"), aka=c.get("aka", []))
    print(f"  Loaded {len(concepts)} concepts")


def populate_references(kg):
    """Core references from demo + lit review."""
    refs = [
        {"id":"pearl-1995","short":"Pearl (1995)","year":1995,
         "full":"Pearl, J. (1995). Causal diagrams for empirical research. Biometrika, 82(4), 669-688.",
         "doi":"10.1093/biomet/82.4.669","type":"paper",
         "proves":["back-door-criterion","d-separation"]},
        {"id":"pearl-2009","short":"Pearl (2009)","year":2009,
         "full":"Pearl, J. (2009). Causality: Models, Reasoning, and Inference (2nd ed.). Cambridge University Press.",
         "doi":"10.1017/CBO9780511803161","type":"book",
         "proves":["do-calculus","front-door-criterion","rung-3"]},
        {"id":"hernan-robins-2020","short":"Hernan & Robins (2020)","year":2020,
         "full":"Hernan, M. A. & Robins, J. M. (2020). Causal Inference: What If. Chapman & Hall/CRC.",
         "type":"book","proves":["positivity","ignorability","sutva"]},
        {"id":"chernozhukov-2018","short":"Chernozhukov et al. (2018)","year":2018,
         "full":"Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1), C1-C68.",
         "doi":"10.1111/ectj.12097","type":"paper",
         "proves":["neyman-orthogonality","double-robustness"]},
        {"id":"vanderweele-ding-2017","short":"VanderWeele & Ding (2017)","year":2017,
         "full":"VanderWeele, T. J. & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. Annals of Internal Medicine, 167(4), 268-274.",
         "doi":"10.7326/M16-2607","type":"paper",
         "proves":["sensitivity-analysis"]},
        {"id":"peters-2016","short":"Peters, Buhlmann & Meinshausen (2016)","year":2016,
         "full":"Peters, J., Buhlmann, P., & Meinshausen, N. (2016). Causal inference by using invariant prediction. JRSS-B, 78(5), 947-1012.",
         "doi":"10.1111/rssb.12167","type":"paper",
         "proves":["invariance-principle"]},
        {"id":"rubin-1974","short":"Rubin (1974)","year":1974,
         "full":"Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. Journal of Educational Psychology, 66(5), 688-701.",
         "doi":"10.1037/h0037350","type":"paper",
         "proves":["ignorability","sutva"]},
        {"id":"berkson-1946","short":"Berkson (1946)","year":1946,
         "full":"Berkson, J. (1946). Limitations of the application of fourfold table analysis to hospital data. Biometrics Bulletin, 2(3), 47-53.",
         "doi":"10.2307/3002000","type":"paper",
         "proves":["collider-bias"]},
        {"id":"lipsitch-2010","short":"Lipsitch, Tchetgen Tchetgen & Cohen (2010)","year":2010,
         "full":"Lipsitch, M., Tchetgen Tchetgen, E., & Cohen, T. (2010). Negative controls: a tool for detecting confounding and bias in observational studies. Epidemiology, 21(3), 383-388.",
         "doi":"10.1097/EDE.0b013e3181d61eeb","type":"paper",
         "proves":["negative-control"]},
        {"id":"spirtes-2000","short":"Spirtes, Glymour & Scheines (2000)","year":2000,
         "full":"Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation, Prediction, and Search (2nd ed.). MIT Press.",
         "type":"book","proves":[]},
        {"id":"hernan-robins-2016","short":"Hernan & Robins (2016)","year":2016,
         "full":"Hernan, M. A. & Robins, J. M. (2016). Using big data to emulate a target trial when a randomized trial is not available. American Journal of Epidemiology, 183(8), 758-764.",
         "doi":"10.1093/aje/kwv254","type":"paper","proves":[]},
        {"id":"fisher-1925","short":"Fisher (1925)","year":1925,
         "full":"Fisher, R. A. (1925). Statistical Methods for Research Workers. Oliver & Boyd.",
         "type":"book","proves":["p-value","maximum-likelihood","type-i-error"]},
        {"id":"neyman-pearson-1933","short":"Neyman & Pearson (1933)","year":1933,
         "full":"Neyman, J. & Pearson, E. S. (1933). On the problem of the most efficient tests of statistical hypotheses. Philosophical Transactions of the Royal Society A, 231, 289-337.",
         "doi":"10.1098/rsta.1933.0009","type":"paper",
         "proves":["type-i-error","statistical-power"]},
        {"id":"frisch-waugh-1933","short":"Frisch & Waugh (1933)","year":1933,
         "full":"Frisch, R. & Waugh, F. V. (1933). Partial time regressions as compared with individual trends. Econometrica, 1(4), 387-401.",
         "doi":"10.2307/1907330","type":"paper","proves":["frisch-waugh-lovell"]},
        # Cross-disciplinary references
        {"id":"hume-1748","short":"Hume (1748)","year":1748,
         "full":"Hume, D. (1748). An Enquiry Concerning Human Understanding. London: A. Millar.",
         "type":"book","proves":["humean-causation"]},
        {"id":"lewis-1973","short":"Lewis (1973)","year":1973,
         "full":"Lewis, D. (1973). Causation. Journal of Philosophy, 70(17), 556-567.",
         "doi":"10.2307/2025310","type":"paper","proves":["counterfactual-theory"]},
        {"id":"woodward-2003","short":"Woodward (2003)","year":2003,
         "full":"Woodward, J. (2003). Making Things Happen: A Theory of Causal Explanation. Oxford University Press.",
         "type":"book","proves":["manipulability-theory"]},
        {"id":"cartwright-1999","short":"Cartwright (1999)","year":1999,
         "full":"Cartwright, N. (1999). The Dappled World: A Study of the Boundaries of Science. Cambridge University Press.",
         "doi":"10.1017/CBO9781139167093","type":"book","proves":["cartwrights-dictum"]},
        {"id":"heckman-1979","short":"Heckman (1979)","year":1979,
         "full":"Heckman, J. J. (1979). Sample selection bias as a specification error. Econometrica, 47(1), 153-161.",
         "doi":"10.2307/1912352","type":"paper","proves":["heckman-selection"]},
        {"id":"granger-1969","short":"Granger (1969)","year":1969,
         "full":"Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. Econometrica, 37(3), 424-438.",
         "doi":"10.2307/1912791","type":"paper","proves":["granger-causality"]},
        {"id":"hill-1965","short":"Hill (1965)","year":1965,
         "full":"Hill, A. B. (1965). The environment and disease: association or causation? Proceedings of the Royal Society of Medicine, 58(5), 295-300.",
         "doi":"10.1177/003591576505800503","type":"paper","proves":["bradford-hill-criteria"]},
        {"id":"greenland-pearl-robins-1999","short":"Greenland, Pearl & Robins (1999)","year":1999,
         "full":"Greenland, S., Pearl, J., & Robins, J. M. (1999). Causal diagrams for epidemiologic research. Epidemiology, 10(1), 37-48.",
         "type":"paper","proves":["confounding-in-epidemiology","d-separation"]},
        {"id":"scholkopf-2021","short":"Scholkopf et al. (2021)","year":2021,
         "full":"Scholkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., & Bengio, Y. (2021). Toward causal representation learning. Proceedings of the IEEE, 109(5), 612-634.",
         "doi":"10.1109/JPROC.2021.3058954","type":"paper","proves":["causal-representation-learning"]},
        {"id":"campbell-stanley-1963","short":"Campbell & Stanley (1963)","year":1963,
         "full":"Campbell, D. T. & Stanley, J. C. (1963). Experimental and Quasi-Experimental Designs for Research. Houghton Mifflin.",
         "type":"book","proves":["campbell-validity","quasi-experiments"]},
        {"id":"baron-kenny-1986","short":"Baron & Kenny (1986)","year":1986,
         "full":"Baron, R. M. & Kenny, D. A. (1986). The moderator-mediator variable distinction in social psychological research. Journal of Personality and Social Psychology, 51(6), 1173-1182.",
         "doi":"10.1037/0022-3514.51.6.1173","type":"paper","proves":["mediation-in-psychology"]},
        {"id":"rosenbaum-rubin-1983","short":"Rosenbaum & Rubin (1983)","year":1983,
         "full":"Rosenbaum, P. R. & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. Biometrika, 70(1), 41-55.",
         "doi":"10.1093/biomet/70.1.41","type":"paper","proves":["propensity-score","ignorability"]},
        {"id":"shpitser-pearl-2006","short":"Shpitser & Pearl (2006)","year":2006,
         "full":"Shpitser, I. & Pearl, J. (2006). Complete identification methods for the causal hierarchy. Journal of Machine Learning Research, 9, 1941-1979.",
         "type":"paper","proves":["id-algorithm","do-calculus"]},
        {"id":"robins-1986","short":"Robins (1986)","year":1986,
         "full":"Robins, J. M. (1986). A new approach to causal inference in mortality studies with sustained exposure periods. Mathematical Modelling, 7(9), 1393-1512.",
         "doi":"10.1016/0270-0255(86)90088-6","type":"paper","proves":["g-methods"]},
        {"id":"imbens-rubin-2015","short":"Imbens & Rubin (2015)","year":2015,
         "full":"Imbens, G. W. & Rubin, D. B. (2015). Causal Inference for Statistics, Social, and Biomedical Sciences. Cambridge University Press.",
         "doi":"10.1017/CBO9781139025751","type":"book","proves":[]},
    ]
    for r in refs:
        proves = r.pop("proves", [])
        kg.run("""
            MERGE (n:Reference {id: $id})
            SET n.short = $short, n.full = $full, n.year = $year
            SET n.type = $type, n.doi = $doi
        """, id=r["id"], short=r["short"], full=r["full"], year=r["year"],
             type=r["type"], doi=r.get("doi"))
        # Link to concepts
        for cid in proves:
            kg.run("""
                MATCH (r:Reference {id: $rid}), (c:Concept {id: $cid})
                MERGE (r)-[:PROVES]->(c)
            """, rid=r["id"], cid=cid)
    print(f"  Loaded {len(refs)} references with PROVES links")


def populate_stations(kg):
    """9 UCL stations from demo content."""
    for s in content["stations"]:
        kg.run("""
            MERGE (n:Station {id: $id})
            SET n.number = $number, n.name = $name, n.emoji = $emoji
            SET n.question = $question, n.output = $output
        """, id=s["id"], number=s["number"], name=s["name"],
             emoji=s.get("emoji",""), question=s.get("question",""), output=s.get("output",""))
    # Sequential PRECEDES edges
    for i in range(len(content["stations"]) - 1):
        a, b = content["stations"][i]["id"], content["stations"][i+1]["id"]
        kg.run("MATCH (a:Station {id:$a}), (b:Station {id:$b}) MERGE (a)-[:PRECEDES]->(b)", a=a, b=b)
    print(f"  Loaded {len(content['stations'])} stations")


def populate_dag(kg):
    """NomNom DAG: variables, edges, absent edges."""
    # Variable nodes
    for v in g.observed:
        role = g.node_roles.get(v, "unknown")
        kg.run("""
            MERGE (n:Variable {id: $id})
            SET n.name = $id, n.role = $role, n.observed = true
        """, id=v, role=role)
    for v in g.latent:
        role = g.node_roles.get(v, "unknown")
        kg.run("""
            MERGE (n:Variable {id: $id})
            SET n.name = $id, n.role = $role, n.observed = false
        """, id=v, role=role)
    # Causal edges
    for src, tgt in g.edges:
        kg.run("""
            MATCH (a:Variable {id: $src}), (b:Variable {id: $tgt})
            MERGE (a)-[:CAUSES]->(b)
        """, src=src, tgt=tgt)
    # Absent edges as explicit NON_CAUSES relationships
    n_absent = 0
    for src, tgt in g.absent_edges:
        kg.run("""
            MATCH (a:Variable {id: $src}), (b:Variable {id: $tgt})
            MERGE (a)-[:NON_CAUSES {falsifiable: true}]->(b)
        """, src=src, tgt=tgt)
        n_absent += 1
    print(f"  Loaded {len(g.observed)+len(g.latent)} variables, {len(g.edges)} edges, {n_absent} absent edges")


def populate_methods(kg):
    """Estimation and discovery methods."""
    methods = [
        {"id":"aipw","name":"AIPW","class":"estimator","rung":2,
         "description":"Augmented Inverse Probability Weighting: doubly-robust, Neyman-orthogonal estimator using cross-fit gradient boosting for nuisances.",
         "requires":["ignorability","positivity","sutva"],"inputs":"adjustment_set","outputs":"ate_estimate"},
        {"id":"dml","name":"Double/Debiased ML","class":"estimator","rung":2,
         "description":"Orthogonal scores + cross-fitting to debias ML nuisance estimates. Achieves sqrt(n)-consistency with data-adaptive nuisances (Chernozhukov et al. 2018).",
         "requires":["ignorability","positivity","sutva","neyman-orthogonality"],"inputs":"adjustment_set","outputs":"ate_estimate"},
        {"id":"ipw","name":"Inverse Probability Weighting","class":"estimator","rung":2,
         "description":"Weights observations by inverse propensity score to create a pseudo-population where treatment is independent of covariates.",
         "requires":["ignorability","positivity"],"inputs":"propensity_model","outputs":"ate_estimate"},
        {"id":"tmle","name":"Targeted Maximum Likelihood","class":"estimator","rung":2,
         "description":"Targets the estimand by fluctuating initial nuisance estimates. Doubly robust and locally efficient (van der Laan & Rose 2011).",
         "requires":["ignorability","positivity","sutva"],"inputs":"adjustment_set","outputs":"ate_estimate"},
        {"id":"pc-algorithm","name":"PC Algorithm","class":"discovery","rung":1,
         "description":"Constraint-based causal discovery: tests conditional independencies, removes edges, orients colliders. Returns Markov equivalence class. Assumes causal sufficiency (no latent confounders).",
         "requires":[],"inputs":"observational_data","outputs":"partial_dag"},
        {"id":"fci-algorithm","name":"FCI Algorithm","class":"discovery","rung":1,
         "description":"Fast Causal Inference: extends PC to allow arbitrary latent confounders. Returns a Partial Ancestral Graph (PAG). More conservative but more honest than PC.",
         "requires":[],"inputs":"observational_data","outputs":"pag"},
        {"id":"ges","name":"Greedy Equivalence Search","class":"discovery","rung":1,
         "description":"Score-based causal discovery: greedily adds/removes edges to maximize BIC. Returns the highest-scoring Markov equivalence class.",
         "requires":[],"inputs":"observational_data","outputs":"equivalence_class"},
    ]
    for m in methods:
        reqs = m.pop("requires", [])
        kg.run("""
            MERGE (n:Method {id: $id})
            SET n.name = $name, n.class = $class, n.rung = $rung
            SET n.description = $description
        """, **m)
        for cid in reqs:
            kg.run("""
                MATCH (m:Method {id: $mid}), (c:Concept {id: $cid})
                MERGE (m)-[:REQUIRES]->(c)
            """, mid=m["id"], cid=cid)
    print(f"  Loaded {len(methods)} methods")


def populate_bridge_levels(kg):
    """6-level math bridge curriculum."""
    levels = [
        {"id":"level-0","level":0,"name":"Foundations","topic":"Probability, Simpson's paradox, FWL",
         "insight":"The data alone cannot tell you whether to condition on 'city' — that decision is causal.",
         "limitation":"Cannot distinguish confounding from causation; rung 1 only."},
        {"id":"level-1","level":1,"name":"Bayesian Inference","topic":"Conjugacy, MCMC, posterior predictive checks",
         "insight":"Bayesian inference updates beliefs about parameters of a fixed observational model — it never leaves rung 1 by itself.",
         "limitation":"Perfect posterior concentrates on the wrong causal effect under confounding."},
        {"id":"level-2","level":2,"name":"Bayesian Networks","topic":"d-separation, Markov condition, structure learning (PC, GES)",
         "insight":"A BN is a joint distribution with a graph; nothing in it yet says 'cause' — any orientation in the Markov equivalence class fits equally well.",
         "limitation":"Markov equivalence: cannot distinguish X→Y from X←Y without additional assumptions."},
        {"id":"level-3","level":3,"name":"The Causal Step","topic":"do-operator, truncated factorization, back-door/front-door, counterfactuals",
         "insight":"do() differs from condition(): P(Y|T) ≠ P(Y|do(T)) exactly by confounding. Front-door criterion applied without checking fails silently.",
         "limitation":"Requires full SCM for counterfactuals (rung 3); DAG alone only reaches rung 2."},
        {"id":"level-4","level":4,"name":"Estimation Theory","topic":"AIPW, DML, double robustness, Neyman orthogonality",
         "insight":"Identification is exact and assumption-driven; estimation is approximate and data-driven — the two error budgets must never be conflated.",
         "limitation":"nD trap: Lasso plug-in bias 0.39 vs DML 0.04; regularization bias contaminates causal estimates in high dimensions."},
        {"id":"level-5","level":5,"name":"Compositional Capstone","topic":"Markov categories, string-diagram surgery, Jacobs-Kissinger-Zanasi",
         "insight":"The back-door criterion is a theorem about the diagram, not an axiom — it can be re-derived as wire surgery in categorical probability.",
         "limitation":"Requires abstract algebra; pedagogical bridge, not a practical tool."},
    ]
    for lv in levels:
        kg.run("""
            MERGE (n:BridgeLevel {id: $id})
            SET n.level = $level, n.name = $name, n.topic = $topic
            SET n.insight = $insight, n.limitation = $limitation
        """, **lv)
    # Prerequisite chain
    for i in range(len(levels) - 1):
        a, b = levels[i]["id"], levels[i+1]["id"]
        kg.run("MATCH (a:BridgeLevel {id:$a}), (b:BridgeLevel {id:$b}) MERGE (a)-[:PREREQUISITE_FOR]->(b)", a=a, b=b)
    print(f"  Loaded {len(levels)} math bridge levels")


def populate_gallery(kg):
    """6 tier-1 gallery cases."""
    cases = [
        {"id":"berkeley-simpson","name":"Berkeley Graduate Admissions","method":"Stratification",
         "result":"Aggregate gap reverses within departments (Bickel et al. 1975)",
         "concepts":["simpsons-paradox","confounding"],"reference":"bickel-1975"},
        {"id":"lalonde-nsw","name":"LaLonde/NSW Job Training","method":"Propensity Score Matching",
         "result":"Naive -$15,205 → PS match +$2,697 vs $1,794 RCT benchmark",
         "concepts":["ignorability","positivity"],"reference":"lalonde-1986"},
        {"id":"card-krueger","name":"Card & Krueger Minimum Wage","method":"Difference-in-Differences",
         "result":"+2.75 FTE in NJ after minimum wage increase",
         "concepts":["rung-2"],"reference":"card-krueger-1994"},
        {"id":"oregon-medicaid","name":"Oregon Medicaid Lottery","method":"Instrumental Variables",
         "result":"LATE on ED visits and depression (Finkelstein et al. 2012)",
         "concepts":["instrumental-variable","late"],"reference":"finkelstein-2012"},
        {"id":"basque-terrorism","name":"Basque Country Terrorism","method":"Synthetic Control",
         "result":"GDP gap opens after 1970 (Abadie & Gardeazabal 2003)",
         "concepts":["rung-2"],"reference":"abadie-2003"},
        {"id":"sachs-proteins","name":"Sachs Protein Signaling","method":"PC/FCI Discovery",
         "result":"24 edges recovered vs 16 interventional ground truth",
         "concepts":["d-separation","rung-1"],"reference":"sachs-2005"},
    ]
    for c in cases:
        concepts = c.pop("concepts", [])
        kg.run("""
            MERGE (n:GalleryCase {id: $id})
            SET n.name = $name, n.method = $method, n.result = $result
        """, **c)
        for cid in concepts:
            kg.run("MATCH (g:GalleryCase {id:$gid}), (c:Concept {id:$cid}) MERGE (g)-[:DEMONSTRATES]->(c)", gid=c["id"], cid=cid)
    print(f"  Loaded {len(cases)} gallery cases")


def populate_principles(kg):
    """6 design principles (P1-P6)."""
    principles = [
        {"id":"P1","number":1,"name":"Assumptions Are First-Class Artifacts",
         "statement":"Every causal claim carries a versioned, inspectable assumption object (DAG + SCM). No claim without its assumptions."},
        {"id":"P2","number":2,"name":"Graph = Single Source of Truth",
         "statement":"Identification, adjustment sets, test suites, and monitoring checks are compiled from the DAG, not hand-maintained."},
        {"id":"P3","number":3,"name":"Sensors + Actuators at Every Stage",
         "statement":"Each station emits quantitative health signals and has defined revision actions — the loop is closeable."},
        {"id":"P4","number":4,"name":"Refutation Is Continuous",
         "statement":"Placebo tests, negative controls, and falsification checks run in CI and production monitoring, not once at publication."},
        {"id":"P5","number":5,"name":"Ground Truth Where Possible",
         "statement":"Development happens against synthetic DGPs with known effects; methods graduate to real data only after passing synthetic acceptance tests."},
        {"id":"P6","number":6,"name":"Ladder Discipline",
         "statement":"Every query is labeled by its rung on Pearl's ladder; rung-2/3 questions with rung-1 machinery require recorded assumptions."},
    ]
    for p in principles:
        kg.run("""
            MERGE (n:Principle {id: $id})
            SET n.number = $number, n.name = $name, n.statement = $statement
        """, **p)
    print(f"  Loaded {len(principles)} design principles")


def populate_relationships(kg):
    """Cross-cutting relationships between different entity types."""
    edges = [
        # Stations use concepts
        ("Station","assume","USES","Concept","d-separation"),
        ("Station","identify","USES","Concept","back-door-criterion"),
        ("Station","model","USES","Concept","aipw"),
        ("Station","model","USES","Concept","neyman-orthogonality"),
        ("Station","model","USES","Concept","double-robustness"),
        ("Station","evaluate","USES","Concept","sensitivity-analysis"),
        ("Station","evaluate","USES","Concept","positivity"),
        ("Station","test","USES","Concept","negative-control"),
        ("Station","test","USES","Concept","ignorability"),
        ("Station","evolve","USES","Concept","invariance-principle"),
        # Principles implemented by stations
        ("Principle","P1","IMPLEMENTED_BY","Station","assume"),
        ("Principle","P2","IMPLEMENTED_BY","Station","identify"),
        ("Principle","P2","IMPLEMENTED_BY","Station","feature"),
        ("Principle","P3","IMPLEMENTED_BY","Station","evolve"),
        ("Principle","P4","IMPLEMENTED_BY","Station","test"),
        ("Principle","P5","IMPLEMENTED_BY","Variable","U"),
        ("Principle","P6","IMPLEMENTED_BY","Concept","rung-1"),
        # Methods use variables from the DAG
        ("Method","aipw","USES_VARIABLE","Variable","T"),
        ("Method","aipw","USES_VARIABLE","Variable","Y"),
        ("Method","aipw","USES_VARIABLE","Variable","W"),
        # Statistics foundations → causal inference
        ("Concept","bayes-rule","PREREQUISITE_FOR","Concept","rung-3"),
        ("Concept","central-limit-theorem","PREREQUISITE_FOR","Concept","confidence-interval"),
        ("Concept","bias-variance-tradeoff","PREREQUISITE_FOR","Concept","regularization"),
        ("Concept","regularization","PREREQUISITE_FOR","Concept","neyman-orthogonality"),
        ("Concept","cross-validation","PREREQUISITE_FOR","Concept","double-robustness"),
        ("Concept","law-of-total-probability","PREREQUISITE_FOR","Concept","back-door-criterion"),
        ("Concept","markov-condition","PREREQUISITE_FOR","Concept","d-separation"),
        ("Concept","exchangeability","PREREQUISITE_FOR","Concept","ignorability"),
        ("Concept","frisch-waugh-lovell","PREREQUISITE_FOR","Method","aipw"),
        # Math bridge connects to statistics
        ("BridgeLevel","level-0","TEACHES","Concept","law-of-total-probability"),
        ("BridgeLevel","level-0","TEACHES","Concept","bayes-rule"),
        ("BridgeLevel","level-0","TEACHES","Concept","frisch-waugh-lovell"),
        ("BridgeLevel","level-1","TEACHES","Concept","maximum-likelihood"),
        ("BridgeLevel","level-1","TEACHES","Concept","bayes-rule"),
        ("BridgeLevel","level-2","TEACHES","Concept","markov-condition"),
        ("BridgeLevel","level-4","TEACHES","Concept","bias-variance-tradeoff"),
        ("BridgeLevel","level-4","TEACHES","Concept","regularization"),
        ("BridgeLevel","level-4","TEACHES","Concept","cross-validation"),
        # Cross-disciplinary bridges: philosophy -> causal foundations
        ("Concept","humean-causation","PREREQUISITE_FOR","Concept","confounding"),
        ("Concept","counterfactual-theory","PREREQUISITE_FOR","Concept","ignorability"),
        ("Concept","manipulability-theory","PREREQUISITE_FOR","Concept","do-calculus"),
        ("Concept","cartwrights-dictum","PREREQUISITE_FOR","Concept","d-separation"),
        # Cross-disciplinary bridges: econometrics -> causal methods
        ("Concept","heckman-selection","PREREQUISITE_FOR","Concept","ignorability"),
        ("Concept","panel-data-methods","PREREQUISITE_FOR","Method","aipw"),
        # Cross-disciplinary bridges: epidemiology -> causal design
        ("Concept","bradford-hill-criteria","PREREQUISITE_FOR","Concept","sensitivity-analysis"),
        ("Concept","target-trial-emulation","PREREQUISITE_FOR","Station","frame"),
        ("Concept","confounding-in-epidemiology","PREREQUISITE_FOR","Concept","confounding"),
        # Cross-disciplinary bridges: CS/AI -> causal ML
        ("Concept","causal-representation-learning","PREREQUISITE_FOR","Concept","neyman-orthogonality"),
        ("Concept","fairness-in-ml","PREREQUISITE_FOR","Concept","sensitivity-analysis"),
        # Cross-disciplinary bridges: psychology -> causal design
        ("Concept","campbell-validity","PREREQUISITE_FOR","Concept","positivity"),
        ("Concept","quasi-experiments","PREREQUISITE_FOR","Method","aipw"),
        # Cross-disciplinary bridges: biostatistics -> causal estimation
        ("Concept","g-methods","PREREQUISITE_FOR","Method","tmle"),
        ("Concept","propensity-score","PREREQUISITE_FOR","Method","ipw"),
        ("Concept","sensitivity-analysis-epi","PREREQUISITE_FOR","Concept","sensitivity-analysis"),
        # Bayesian → causal bridge
        ("Concept","bayesian-marginalization","PREREQUISITE_FOR","Concept","back-door-criterion"),
        ("Concept","bayesian-marginalization","PREREQUISITE_FOR","Concept","conditioning-vs-intervening"),
        ("Concept","conditioning-vs-intervening","PREREQUISITE_FOR","Concept","do-calculus"),
        ("Concept","truncated-factorization","PREREQUISITE_FOR","Concept","back-door-criterion"),
        ("Concept","truncated-factorization","PREREQUISITE_FOR","Concept","front-door-criterion"),
        ("Concept","id-algorithm","PREREQUISITE_FOR","Concept","do-calculus"),
        # Math bridge links
        ("BridgeLevel","level-1","TEACHES","Concept","bayesian-marginalization"),
        ("BridgeLevel","level-3","TEACHES","Concept","conditioning-vs-intervening"),
        ("BridgeLevel","level-3","TEACHES","Concept","truncated-factorization"),
        # Reference links
        ("Reference","pearl-2009","PROVES","Concept","truncated-factorization"),
        ("Reference","pearl-2009","PROVES","Concept","front-door-criterion"),
        ("Reference","shpitser-pearl-2006","PROVES","Concept","id-algorithm"),
    ]
    for src_type, src_id, rel, tgt_type, tgt_id in edges:
        kg.run(f"""
            MATCH (a:{src_type} {{id: $src_id}}), (b:{tgt_type} {{id: $tgt_id}})
            MERGE (a)-[:{rel}]->(b)
        """, src_id=src_id, tgt_id=tgt_id)
    print(f"  Loaded {len(edges)} cross-cutting relationships")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    dry = "--dry" in sys.argv
    kg = KG(dry=dry)

    if not dry:
        kg.clear()

    populate_concepts(kg)
    populate_references(kg)
    populate_stations(kg)
    populate_dag(kg)
    populate_methods(kg)
    populate_bridge_levels(kg)
    populate_gallery(kg)
    populate_principles(kg)
    populate_relationships(kg)

    # Summary
    if not dry:
        with kg.driver.session() as s:
            r = s.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC")
            print("\n  Node counts:")
            for rec in r:
                print(f"    {rec['label']}: {rec['cnt']}")
            r2 = s.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC")
            print("  Relationship counts:")
            for rec in r2:
                print(f"    {rec['rel']}: {rec['cnt']}")

    kg.close()
    print("\nDone.")
