import matplotlib.pyplot as plt
import networkx as nx

fig, ax = plt.subplots(1, 1, figsize=(14, 9))

G = nx.DiGraph()
nodes = {
    "Z": "Z\njitter", "rain": "rain", "payday": "payday",
    "weekend": "wknd", "W": "W\napp-use", "U": "U\nhunger",
    "T": "T\nnotify", "M": "M\nopen", "Y": "Y\norder",
    "S": "S\nengage", "NC": "NC\nbattery", "coupon": "D\ncoupon",
    "segment": "seg", "loyalty": "loyalty"
}
G.add_nodes_from(nodes.keys())

causal = [("T", "Y")]
confound = [("U", "W"), ("W", "T"), ("U", "Y"), ("U", "M"), ("U", "NC")]
instrument = [("Z", "T")]
mediator = [("T", "M"), ("M", "Y")]
collider = [("T", "S"), ("Y", "S")]
other = [("weekend", "T"), ("weekend", "Y"), ("rain", "T"), ("rain", "Y"),
         ("payday", "T"), ("payday", "Y"), ("segment", "Y"),
         ("loyalty", "coupon"), ("coupon", "Y")]

pos = {
    "U": (0.05, 0.50), "W": (0.22, 0.50),
    "Z": (0.22, 0.80), "rain": (0.22, 0.95), "payday": (0.22, 0.05),
    "weekend": (0.22, 0.20), "T": (0.42, 0.50), "M": (0.62, 0.50),
    "Y": (0.82, 0.50), "S": (0.62, 0.15), "NC": (0.05, 0.15),
    "coupon": (0.62, 0.80), "loyalty": (0.42, 0.80), "segment": (0.42, 0.95),
}

styles = [
    (causal, "#2ecc40", 4.0, "solid", "causal target"),
    (confound, "#e74c3c", 2.0, "dashed", "confounding"),
    (instrument, "#3498db", 2.5, "dotted", "instrument"),
    (mediator, "#8e44ad", 2.0, "dashed", "mediator"),
    (collider, "#e67e22", 2.0, "dotted", "collider"),
    (other, "#95a5a6", 1.5, "solid", "other"),
]
for edge_list, color, width, style, label in styles:
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=color,
                           width=width, style=style, ax=ax,
                           connectionstyle="arc3,rad=0.08",
                           min_source_margin=18, min_target_margin=18)

colors = {"U": "#e8e8e8", "T": "#2ecc40", "Y": "#2ecc40",
          "S": "#e67e22", "M": "#8e44ad", "Z": "#3498db",
          "NC": "#e74c3c", "W": "#f39c12"}
node_colors = [colors.get(n, "#d5dbdb") for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900,
                       edgecolors="#2c3e50", linewidths=1.5, ax=ax)
nx.draw_networkx_labels(G, pos, {n: nodes[n] for n in G.nodes()},
                        font_size=8, font_weight="bold", ax=ax)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="#2ecc40", lw=3, label="Causal effect (T -> Y)"),
    Line2D([0], [0], color="#e74c3c", lw=2, ls="dashed", label="Confounding (U path)"),
    Line2D([0], [0], color="#3498db", lw=2, ls="dotted", label="Instrument (Z)"),
    Line2D([0], [0], color="#8e44ad", lw=2, ls="dashed", label="Mediator (M)"),
    Line2D([0], [0], color="#e67e22", lw=2, ls="dotted", label="Collider (S)"),
    Line2D([0], [0], color="#95a5a6", lw=1.5, label="Other confounders"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=7, framealpha=0.9)
ax.set_title("NomNom Eats -- Causal DAG\nDo push notifications (T) cause orders (Y)?",
             fontsize=13, fontweight="bold", pad=15)
ax.axis("off")
plt.tight_layout()
plt.show()
