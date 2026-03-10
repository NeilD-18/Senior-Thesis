#!/usr/bin/env python3
"""Create Week 8 Airbnb regression summary plots."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


MODEL_ORDER = ["rf", "linear", "xgb"]
MODEL_LABEL = {
    "rf": "Random Forest (Reg)",
    "linear": "Linear (Ridge fallback)",
    "xgb": "XGBoost (Reg)",
}
MODEL_COLOR = {
    "rf": "#2E86AB",
    "linear": "#44AF69",
    "xgb": "#E94F37",
}


def _load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def load_regression_stability(week8_dir: Path, corruption: str):
    if corruption == "noise":
        dir_map = {
            "rf": week8_dir / "airbnb_noise_rf" / "stability_summary.json",
            "linear": week8_dir / "airbnb_noise_linear" / "stability_summary.json",
            "xgb": week8_dir / "airbnb_noise_xgb" / "stability_summary.json",
        }
    elif corruption == "missingness":
        dir_map = {
            "rf": week8_dir / "airbnb_missingness_rf" / "stability_summary.json",
            "linear": week8_dir / "airbnb_missingness_linear" / "stability_summary.json",
            "xgb": week8_dir / "airbnb_missingness_xgb" / "stability_summary.json",
        }
    else:
        raise ValueError(f"Unknown corruption: {corruption}")

    out = {}
    for model, path in dir_map.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing summary: {path}")
        rows = _load_json(path)
        out[model] = sorted(rows, key=lambda r: r["severity"])
    return out


def plot_regression_comparison(data_by_model, output_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    metrics = [("test_rmse", "RMSE"), ("test_mae", "MAE")]

    for ax, (metric_key, label) in zip(axes, metrics):
        for model in MODEL_ORDER:
            rows = data_by_model[model]
            mean_key = f"{metric_key}_mean"
            std_key = f"{metric_key}_std"
            rows_used = [r for r in rows if mean_key in r]
            if not rows_used:
                continue
            severities = [r["severity"] for r in rows_used]
            means = [r[mean_key] for r in rows_used]
            stds = [r.get(std_key, 0.0) for r in rows_used]
            ax.errorbar(
                severities,
                means,
                yerr=stds,
                marker="o",
                linewidth=2,
                capsize=3,
                alpha=0.9,
                color=MODEL_COLOR[model],
                label=MODEL_LABEL[model],
            )
        ax.set_xlabel("Corruption Severity")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, max([max([r["severity"] for r in data_by_model[m]]) for m in MODEL_ORDER]) + 0.02])

    axes[0].legend(loc="best", framealpha=0.9)
    fig.suptitle(title, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Week 8 Airbnb summary plots")
    parser.add_argument("--week8-dir", type=str, default="outputs/week8_20260305")
    args = parser.parse_args()

    week8_dir = Path(args.week8_dir)
    noise_data = load_regression_stability(week8_dir, corruption="noise")
    miss_data = load_regression_stability(week8_dir, corruption="missingness")

    plot_regression_comparison(
        noise_data,
        week8_dir / "airbnb_noise_model_comparison.png",
        title="Airbnb Additive Noise: Model Comparison (mean +/- std)",
    )
    plot_regression_comparison(
        miss_data,
        week8_dir / "airbnb_missingness_model_comparison.png",
        title="Airbnb Missingness: Model Comparison (mean +/- std)",
    )


if __name__ == "__main__":
    main()

