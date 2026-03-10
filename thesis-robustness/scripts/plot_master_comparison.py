#!/usr/bin/env python3
"""Generate a cross-experiment comparison figure from master summary stats."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _load(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def plot_master(summary, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1: Adult AUROC deltas across corruption families.
    adult = summary["adult"]
    corr_order = ["noise", "missingness", "imbalance"]
    corr_label = {
        "noise": "Noise",
        "missingness": "Missingness",
        "imbalance": "Imbalance",
    }
    adult_models = ["RF", "XGB", "SVM-RBF"]
    model_color = {
        "RF": "#2E86AB",
        "XGB": "#E94F37",
        "SVM-RBF": "#44AF69",
    }

    ax = axes[0]
    x = np.arange(len(corr_order))
    width = 0.24
    for i, model in enumerate(adult_models):
        vals = [adult[c][model]["delta_test_auroc"] * 100 for c in corr_order]
        ax.bar(x + (i - 1) * width, vals, width=width, label=model, color=model_color[model], alpha=0.92)
    ax.axhline(0, color="#666666", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([corr_label[c] for c in corr_order])
    ax.set_ylabel("Delta AUROC (pp)\n(worst - clean)")
    ax.set_title("Adult: Corruption Sensitivity")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", framealpha=0.9)

    # Panel 2: IMDB in-domain vs out-of-domain absolute AUROC.
    imdb_shift = summary["imdb_domain_shift"]
    text_models = ["Linear SVM", "Random Forest", "XGBoost"]
    imdb_auroc = [imdb_shift[m]["imdb_auroc"] for m in text_models]
    amazon_auroc = [imdb_shift[m]["amazon_auroc"] for m in text_models]
    x = np.arange(len(text_models))
    width = 0.32
    ax = axes[1]
    bars_imdb = ax.bar(x - width / 2, imdb_auroc, width=width, color="#2E86AB", alpha=0.9, label="IMDB (in-domain)")
    bars_amazon = ax.bar(x + width / 2, amazon_auroc, width=width, color="#E94F37", alpha=0.9, label="Amazon (out-of-domain)")
    for i, (v_imdb, v_amazon) in enumerate(zip(imdb_auroc, amazon_auroc)):
        drop = (v_imdb - v_amazon) * 100
        mid_y = (v_imdb + v_amazon) / 2
        ax.annotate(
            f"-{drop:.1f} pp",
            xy=(x[i] + width * 0.6, mid_y),
            fontsize=8.5, fontweight="bold", color="#333333",
            ha="left", va="center",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(text_models, rotation=10)
    ax.set_ylabel("Test AUROC")
    ax.set_ylim([0.78, 0.96])
    ax.set_title("IMDB: Domain Shift Penalty")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9)

    # Panel 3: Airbnb RMSE increases for noise vs missingness.
    airbnb = summary["airbnb"]
    corr_order = ["noise", "missingness"]
    corr_label = {
        "noise": "Noise",
        "missingness": "Missingness",
    }
    reg_models = ["Random Forest", "Linear (Ridge fallback)", "XGBoost"]
    reg_color = {
        "Random Forest": "#2E86AB",
        "Linear (Ridge fallback)": "#44AF69",
        "XGBoost": "#E94F37",
    }
    x = np.arange(len(corr_order))
    width = 0.24
    ax = axes[2]
    for i, model in enumerate(reg_models):
        vals = [airbnb[c][model]["delta_test_rmse"] for c in corr_order]
        ax.bar(x + (i - 1) * width, vals, width=width, label=model, color=reg_color[model], alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels([corr_label[c] for c in corr_order])
    ax.set_ylabel("Delta RMSE (worst - clean)")
    ax.set_title("Airbnb Regression: Error Inflation")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", framealpha=0.9)

    fig.suptitle("Master Robustness Comparison Across All Experiments", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate master cross-experiment comparison figure")
    parser.add_argument(
        "--summary-json",
        type=str,
        default="outputs/master_20260309/master_summary_stats.json",
        help="Path to master summary stats JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/master_20260309/master_combined_comparison.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    summary = _load(Path(args.summary_json))
    plot_master(summary, Path(args.output))


if __name__ == "__main__":
    main()

