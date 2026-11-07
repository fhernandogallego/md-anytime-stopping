"""Reproduce Figure 1 (paired MAE contrasts by cohort) from results/paired_contrasts.csv."""
import csv, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(root / "results" / "paired_contrasts.csv")))
cohorts = list(dict.fromkeys(r["cohort"] for r in rows))
n = {r["cohort"]: r["n_systems"] for r in rows}
series = [("ridge_minus_persistence", "Ridge − persistence", "#2a78d6", "o"),
          ("gru_minus_persistence", "GRU − persistence", "#eb6834", "s"),
          ("gru_minus_ridge", "GRU − Ridge", "#1baf7a", "^")]
offsets = [0.22, 0.0, -0.22]

plt.rcParams.update({"font.size": 9, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.axvline(0, color="#8a8a85", lw=1, zorder=0)
for (key, label, color, marker), off in zip(series, offsets):
    for i, c in enumerate(cohorts):
        r = next(x for x in rows if x["cohort"] == c and x["contrast"] == key)
        y = len(cohorts) - 1 - i + off
        ax.plot([float(r["ci95_low_A"]), float(r["ci95_high_A"])], [y, y], color=color, lw=2, solid_capstyle="round")
        ax.plot(float(r["delta_mae_A"]), y, marker, color=color, ms=6, mec="white", mew=1, label=label if i == 0 else None)
labels = [f"{c}\n({n[c]}{' systems' if i == 0 else ''})" for i, c in enumerate(cohorts)]
ax.set_yticks(range(len(cohorts)))
ax.set_yticklabels(labels[::-1])
ax.set_xlabel(r"Paired $\Delta$MAE (Å), first model minus second")
ax.grid(axis="x", color="#e6e5e0", lw=0.6)
ax.set_axisbelow(True)
fig.legend(frameon=False, loc="upper center", ncol=3, handlelength=1.2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(root / "paper" / "figures" / "fig1_paired_contrasts.pdf")
