"""Render the CS2 model comparison from the frozen experiment results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = json.loads((ROOT / "research/experiments/cs2cd_v1/test_results.json").read_text())
    cohort = result["cohorts"]["all"]
    names = ["lstm5", "lstm39", "transformer39", "tcn39", "lgbm39_leaves15", "selected"]
    labels = ["LSTM / 5 channels", "LSTM / 39 channels", "Patch Transformer", "Temporal CNN", "Boosted trees", "Selected blend"]
    ink, muted = "#202020", "#777777"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(8.0, 2.65), facecolor="white")
    fig.subplots_adjust(left=.255, right=.90, top=.96, bottom=.21)
    for i, (name, label) in enumerate(zip(names, labels)):
        auc = cohort["models"][name]["roc_auc"]
        low, high = cohort["bootstrap"]["auc_95"][name]
        color = ink if name == "selected" else muted
        ax.errorbar(auc, i, xerr=np.array([[auc-low], [high-auc]]),
                    fmt="D" if name == "selected" else "o", color=color,
                    markerfacecolor=ink if name == "selected" else "white",
                    markersize=5 if name == "selected" else 4.5,
                    linewidth=1.05, capsize=2.5, markeredgewidth=1.0)
        ax.text(1.004, i, f"{auc:.3f}", va="center", color=ink,
                weight="bold" if name == "selected" else "normal", clip_on=False)
    ax.set_yticks(range(len(names)), labels, color=ink)
    ax.set_ylim(5.6, -.6)
    ax.set_xlim(.85, 1.0)
    ax.set_xticks([.85, .90, .95, 1.0], ["0.85", "0.90", "0.95", "1.00"])
    ax.set_xlabel("Test ROC-AUC", color=muted, fontsize=9)
    ax.tick_params(axis="both", length=0, labelsize=9)
    ax.tick_params(axis="y", pad=14)
    ax.grid(axis="x", color="#dddddd", linewidth=.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    dest = ROOT / "docs/assets"
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / "cs2_comparison.svg", facecolor="white")
    fig.savefig(dest / "cs2_comparison.png", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
