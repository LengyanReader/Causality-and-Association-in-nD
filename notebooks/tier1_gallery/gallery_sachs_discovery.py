"""Tier-1 Gallery #5 — Sachs protein-signaling causal discovery (Sachs et al.
2005, Science; LR section 6, section 13.1).

The gold-standard validation of causal discovery: 11 proteins measured under
9 experimental conditions including specific inhibitions and activations.
The network learned from purely observational data is compared to the
consensus interventional pathway validated by western blots.

Uses the Sachs dataset bundled with `causal-learn` (Zheng et al. 2024).

Run:  python notebooks/tier1_gallery/gallery_sachs_discovery.py
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from causallearn.utils.Dataset import load_dataset
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.graph.Endpoint import Endpoint
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "causal-learn", "-q"])
    from causallearn.utils.Dataset import load_dataset
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.graph.Endpoint import Endpoint


# Consensus interventional edges (Sachs et al. 2005 Fig. 2).
# Protein names as they appear in the causal-learn dataset (lowercase).
TRUTH = {
    ("pip3", "pip2"), ("plc", "pip2"), ("pip3", "plc"),
    ("pkc", "pka"), ("pkc", "p38"), ("pkc", "jnk"), ("pkc", "raf"),
    ("pka", "raf"), ("pka", "mek"), ("pka", "p38"), ("pka", "jnk"),
    ("raf", "mek"), ("mek", "erk"), ("erk", "akt"),
    ("pka", "akt"), ("jnk", "akt"),
}


def all_edges(cg_or_graph, name_map: dict[str, str] | None = None) -> set[tuple[str, str]]:
    """Return all adjacencies (any edge type), not just directed ones."""
    g = cg_or_graph.G if hasattr(cg_or_graph, 'G') else cg_or_graph
    edges = set()
    for e in g.get_graph_edges():
        n1, n2 = e.get_node1().get_name(), e.get_node2().get_name()
        if name_map:
            n1 = name_map.get(n1, n1); n2 = name_map.get(n2, n2)
        edges.add((n1, n2))
    return edges


def directed_edges(cg_or_graph, name_map: dict[str, str] | None = None) -> set[tuple[str, str]]:
    """Return only fully-directed edges (Tail->Arrow or Arrow<-Tail)."""
    g = cg_or_graph.G if hasattr(cg_or_graph, 'G') else cg_or_graph
    edges = set()
    for e in g.get_graph_edges():
        ep1, ep2 = e.get_endpoint1(), e.get_endpoint2()
        n1, n2 = e.get_node1().get_name(), e.get_node2().get_name()
        if name_map:
            n1 = name_map.get(n1, n1)
            n2 = name_map.get(n2, n2)
        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            edges.add((n1, n2))
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            edges.add((n2, n1))
    return edges


def precision(est: set, truth: set) -> float:
    return len(est & truth) / len(est) if est else 0.0


def recall(est: set, truth: set) -> float:
    return len(est & truth) / len(truth) if truth else 0.0


def main() -> None:
    print("=" * 64)
    print("Sachs protein-signaling network — causal discovery")
    print("=" * 64)

    data_np, var_names = load_dataset('sachs')
    df = pd.DataFrame(data_np, columns=var_names)
    # PC/FCI get the raw array; they assign generic X1..X11 names.
    # Map those back to the actual column names.
    x_to_name = {f"X{i+1}": v for i, v in enumerate(var_names)}

    print(f"dataset : {len(df)} cells x {len(df.columns)} proteins")
    print(f"variables: {var_names}")
    print(f"ground-truth edges (interventionally validated) : {len(TRUTH)}")

    # PC — ignores latent confounders
    print("\n--- PC algorithm (Fisher's Z, alpha=0.01) ---")
    cg_pc = pc(data_np, alpha=0.01)
    e_pc = directed_edges(cg_pc, x_to_name)
    p_pc, r_pc = precision(e_pc, TRUTH), recall(e_pc, TRUTH)

    # FCI — handles latent confounders
    print("--- FCI algorithm ---")
    cg_fci = fci(data_np, alpha=0.01)
    e_fci = directed_edges(cg_fci[0] if isinstance(cg_fci, tuple) else cg_fci, x_to_name)
    e_fci_all = all_edges(cg_fci[0] if isinstance(cg_fci, tuple) else cg_fci, x_to_name)
    p_fci, r_fci = precision(e_fci, TRUTH), recall(e_fci, TRUTH)

    print(f"\nPC  : {len(e_pc)} edges, precision {p_pc:.0%}, recall {r_pc:.0%}")
    print(f"FCI : {len(e_fci)} directed + {len(e_fci_all)-len(e_fci)} ambiguous "
          f"= {len(e_fci_all)} total; precision {p_fci:.0%}, recall {r_fci:.0%}")
    assert len(e_pc) >= 5, "PC should recover non-trivial structure"
    # FCI is more conservative than PC: it catches latent confounding but
    # marks many edges as ambiguous, resulting in fewer fully-directed edges
    print(f"\nPC passed : {len(e_pc & TRUTH)}/{len(TRUTH)} truth edges recovered")
    print(f"FCI found : {len(e_fci & TRUTH)}/{len(TRUTH)} truth edges as directed")
    print(f"FCI total : {len(e_fci_all & TRUTH)}/{len(TRUTH)} truth edges as any adjacency")

    print("\nUCL station 8 (EVOLVE/Discovery): the interventional ground truth")
    print("is the rare luxury — but in this gallery script it exists (western")
    print("blots), so the discovery algorithms can be SCORED. The PC recall cap")
    print("is the fingerprint of latent confounding; FCI acknowledges it.")


if __name__ == "__main__":
    main()
