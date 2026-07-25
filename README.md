# Causality-and-Association-in-nD

Discussions on Causality, World Modeling, and Spatial-Motion Analysis

A living research + practice project on causality science: a comprehensive
literature review, a self-evolving causal workflow (**UCL — Universal Causal
Loop**), and a ground-truth reference use case (**NomNom Eats**) that exercises
every component of the workflow.

## Documents (`ref/`)

- [`ref/literature_review.md`](ref/literature_review.md) — theory base: first
  principles, frameworks, identification, estimation, discovery, workflow,
  tools, practice projects, interdisciplinary bridges, full bibliography.
- [`ref/causal_workflow_implementation_plan.md`](ref/causal_workflow_implementation_plan.md)
  — the implementation plan: UCL stations, self-evolution meta-loop, loop/graph
  engineering integration, NomNom use case, math bridge, roadmap.

## Quickstart

```powershell
conda env create -f environment.yml   # or: conda create -n causality-nd python=3.12 numpy pandas scipy scikit-learn networkx pytest
conda activate causality-nd

# end-to-end UCL pass on the static regime (writes runs/static_pass_report.json)
python notebooks/nomnom_endtoend/run_static_pass.py

# causal CI test pyramid (unit layer)
python -m pytest causal_ci/unit -q
```

## Layout

| Path | Contents |
|---|---|
| `ref/` | Living documents (literature review, implementation plan) |
| `nomnom/` | Ground-truth DGP simulator + regimes + causal graph |
| `ucl/` | Universal Causal Loop: contracts, stations, engine, graph compiler |
| `causal_ci/` | Test pyramid: unit / integration / property / regression |
| `notebooks/` | End-to-end runners, math bridge, Tier-1 gallery |
| `loops/` | Loop-engineering integration (event-driven runners) — Phase 4 |
| `runs/` | Generated artifact chains (JSON reports) |

## Status

- **P0 scaffolding** ✅ — repo layout, artifact contracts
- **P1 first loop pass** ✅ — NomNom DGP with known ground truth; UCL stations
  0–7 (frame → assume → identify → data → feature → model → evaluate → test)
  all green; cross-fit AIPW recovers the true ATE; refuters pass on the valid
  pipeline and fire on planted bias
- **P2 causal CI pyramid** ✅ — 32 tests across all four layers:
  unit (DGP sanity, ground-truth recovery, refuters), integration
  (graph-compiled tests: deleting a back-door edge auto-changes the adjustment
  set and turns the pinned-bias test red), property (loop invariants across
  seeds/regimes), regression (pinned ground truths & pipeline outputs)
- **P3 discovery & evolution** ✅ — EVOLVE station: mechanism-stability monitor
  detects the holiday drift and localizes it to the notification→app-open
  mechanism (M), confirms the Y-mechanism invariant, flags the seasonal rain
  shift as marginal; actuator re-run recovers the holiday truth
  (+0.1913 vs +0.1910). See `notebooks/nomnom_endtoend/run_holiday_episode.py`
- **P4 loop/graph engineering integration** — planned
- **P5 math bridge + Tier-1 gallery** — in progress: Level 0 (foundations:
  Bayes by simulation, Simpson's paradox, FWL) and Level 1 (Bayesian inference:
  conjugacy, MH-from-scratch, and the bridge insight that posteriors never
  leave rung 1) done; Levels 2–5 and Tier-1 gallery next
