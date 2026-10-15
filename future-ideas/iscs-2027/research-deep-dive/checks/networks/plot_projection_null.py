"""Standalone figure using saved aggregate results; no inference or raw text."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).absolute().parent
data = json.loads((here / "results.json").read_text())
stored = json.loads((here.parents[4] / "papers/hierarchical-memory/experiments/results/"
                     "hierarchy_v2/heaps_null_model.json").read_text())["metaconcept_network"]
draws = [draw for draw in data["projection_null"]["draws"]
         if draw["trades_per_episode"] == 40]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.4))
labels = ["Observed", "ER null\n(stored mean)", "Bipartite null\n(16 draws)"]
for ax, metric, er_key, ylabel, limits in [
    (axes[0], "clustering", "C_rand", "Mean clustering", (0, .45)),
    (axes[1], "mean_path", "L_rand", "Mean shortest path", (1.8, 2.55)),
]:
    observed = data["observed_projection"][metric]
    values = np.array([draw[metric] for draw in draws])
    ax.axhline(observed, color="#b85b31", linewidth=1, alpha=.45)
    ax.scatter([0], [observed], color="#b85b31", marker="D", s=60, zorder=4)
    ax.scatter([1], [stored[er_key]], color="#687785", marker="s", s=60, zorder=4)
    ax.scatter(2 + np.linspace(-.12, .12, len(values)), values,
               color="#28758b", alpha=.65, s=24, zorder=3)
    ax.plot([1.8, 2.2], [values.mean(), values.mean()], color="#173d4c", lw=2, zorder=4)
    for x, value in [(0, observed), (1, stored[er_key]), (2, values.mean())]:
        ax.annotate(f"{value:.3f}", (x, value), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=11)
    ax.set_xticks([0, 1, 2], labels)
    ax.set_xlim(-.45, 2.45)
    ax.set_ylim(limits)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e7e7e7", linewidth=.8)
    ax.set_axisbelow(True)
fig.suptitle("Concept projection: small-world comparison depends on the null", fontsize=15, y=.98)
fig.text(.5, .895, "Observed/null clustering-to-path ratio: 6.57 against ER; 0.70 against the bipartite null",
         ha="center", fontsize=11)
fig.text(.08, .09,
         "ER: stored analysis's 100-graph mean. Bipartite: fixed episode and concept degrees; 40 trades per episode.\n"
         "Dots are diagnostic runs, not a confidence interval. Mixing is not proved. Largest components: observed 499, null 500.\n"
         "Projected edges: observed 5,935; bipartite mean 8,943. The bipartite null preserves incidence degrees, not projected density.",
         fontsize=9, color="#46515a", va="top")
fig.subplots_adjust(top=.79, bottom=.28, left=.08, right=.98, wspace=.3)
for extension in ("png", "svg"):
    fig.savefig(here / f"projection-null.{extension}", dpi=180, facecolor="white")
