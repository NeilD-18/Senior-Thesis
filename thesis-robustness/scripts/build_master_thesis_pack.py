#!/usr/bin/env python3
"""Build thesis-ready master figures and tables from canonical outputs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_run_coverage(summary, output_path: Path):
    groups = [
        ("Week 6 Adult", ["week6_"]),
        ("Week 7 Dropout", ["week7_dropout_"]),
        ("Week 7 Domain", ["week7_domain_"]),
        ("Week 8 Airbnb", ["week8_"]),
    ]
    got_vals, exp_vals, labels = [], [], []
    for label, prefixes in groups:
        g, e = 0, 0
        for key, item in summary["coverage"].items():
            if any(key.startswith(p) for p in prefixes):
                g += item["got"]
                e += item["expected"]
        labels.append(label)
        got_vals.append(g)
        exp_vals.append(e)

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.bar(x - width / 2, exp_vals, width=width, label="Expected", color="#95A5A6")
    ax.bar(x + width / 2, got_vals, width=width, label="Found", color="#2E86AB")
    for i, (g, e) in enumerate(zip(got_vals, exp_vals)):
        ax.text(i + width / 2, g + 2, f"{g}/{e}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Run files")
    ax.set_title("Master Run Coverage Audit")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_adult_auroc_curves(outputs_root: Path, output_path: Path):
    cfg = [
        ("Noise", "week6_rerun/adult_noise_", 1.0),
        ("Missingness", "week6_rerun/adult_missingness_", 0.5),
        ("Imbalance", "week6_rerun/adult_imbalance_", 0.199),
    ]
    model_map = {
        "rf": ("Random Forest", "#2E86AB"),
        "xgb": ("XGBoost", "#E94F37"),
        "svm": ("SVM-RBF", "#44AF69"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for ax, (title, prefix, xmax) in zip(axes, cfg):
        for key, (label, color) in model_map.items():
            rows = load_json(outputs_root / f"{prefix}{key}/stability_summary.json")
            rows = sorted(rows, key=lambda r: r["severity"])
            x = [r["severity"] for r in rows]
            y = [r["test_auroc_mean"] for r in rows]
            yerr = [r.get("test_auroc_std", 0.0) for r in rows]
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=2, color=color, alpha=0.9, label=label)
        ax.set_title(f"Adult {title}")
        ax.set_xlabel("Severity")
        ax.set_ylabel("Test AUROC")
        ax.set_xlim([-0.01, xmax + 0.01])
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", framealpha=0.9)
    fig.suptitle("Adult: AUROC Degradation Curves (mean +/- std)", fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_imdb_token_dropout(outputs_root: Path, output_path: Path):
    model_cfg = [
        ("linear_svm", "Linear SVM", "#44AF69"),
        ("rf", "Random Forest", "#2E86AB"),
        ("xgb", "XGBoost", "#E94F37"),
    ]
    metric_cfg = [
        ("test_accuracy", "Accuracy"),
        ("test_f1", "F1"),
        ("test_auroc", "AUROC"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for ax, (metric_key, metric_label) in zip(axes, metric_cfg):
        for model_key, model_label, color in model_cfg:
            rows = load_json(outputs_root / f"week7_rerun_20260305/imdb_token_dropout_{model_key}/stability_summary.json")
            rows = sorted(rows, key=lambda r: r["severity"])
            x = [r["severity"] for r in rows]
            y = [r[f"{metric_key}_mean"] for r in rows]
            yerr = [r.get(f"{metric_key}_std", 0.0) for r in rows]
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=2, color=color, alpha=0.9, label=model_label)
        ax.set_title(f"IMDB Token Dropout: {metric_label}")
        ax.set_xlabel("Severity")
        ax.set_ylabel(metric_label)
        ax.set_xlim([-0.01, 0.51])
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", framealpha=0.9)
    fig.suptitle("IMDB Token Dropout Curves (mean +/- std)", fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_imdb_domain_shift(summary, output_path: Path):
    models = ["Linear SVM", "Random Forest", "XGBoost"]
    metrics = [
        ("drop_accuracy_pp", "Accuracy drop"),
        ("drop_f1_pp", "F1 drop"),
        ("drop_auroc_pp", "AUROC drop"),
    ]
    x = np.arange(len(models))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    colors = ["#2E86AB", "#E94F37", "#F39C12"]
    for i, (key, label) in enumerate(metrics):
        vals = [summary["imdb_domain_shift"][m][key] for m in models]
        ax.bar(x + (i - 1) * width, vals, width=width, label=label, color=colors[i], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Drop (percentage points)")
    ax.set_title("IMDB -> Amazon Domain Shift Penalty by Metric")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_airbnb_absolute(outputs_root: Path, output_path: Path):
    model_cfg = [
        ("rf", "Random Forest", "#2E86AB"),
        ("linear", "Linear (Ridge fallback)", "#44AF69"),
        ("xgb", "XGBoost", "#E94F37"),
    ]
    panel_cfg = [
        ("noise", "test_rmse", "RMSE"),
        ("noise", "test_mae", "MAE"),
        ("missingness", "test_rmse", "RMSE"),
        ("missingness", "test_mae", "MAE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    for ax, (corr, metric, ylab) in zip(axes.flat, panel_cfg):
        for mkey, mlabel, color in model_cfg:
            rows = load_json(outputs_root / f"week8_20260305/airbnb_{corr}_{mkey}/stability_summary.json")
            rows = sorted(rows, key=lambda r: r["severity"])
            x = [r["severity"] for r in rows]
            y = [r[f"{metric}_mean"] for r in rows]
            yerr = [r.get(f"{metric}_std", 0.0) for r in rows]
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=2, color=color, alpha=0.9, label=mlabel)
        ax.set_title(f"Airbnb {corr.capitalize()} - {ylab}")
        ax.set_xlabel("Severity")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best", framealpha=0.9)
    fig.suptitle("Airbnb Regression Degradation Curves (mean +/- std)", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_airbnb_relative(outputs_root: Path, output_path: Path):
    model_cfg = [
        ("rf", "Random Forest", "#2E86AB"),
        ("linear", "Linear (Ridge fallback)", "#44AF69"),
        ("xgb", "XGBoost", "#E94F37"),
    ]
    corr_cfg = [("noise", "Noise"), ("missingness", "Missingness")]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4))
    for ax, (corr_key, corr_label) in zip(axes, corr_cfg):
        for mkey, mlabel, color in model_cfg:
            rows = load_json(outputs_root / f"week8_20260305/airbnb_{corr_key}_{mkey}/stability_summary.json")
            rows = sorted(rows, key=lambda r: r["severity"])
            base = rows[0]["test_rmse_mean"]
            x = [r["severity"] for r in rows]
            y = [100.0 * (r["test_rmse_mean"] - base) / base for r in rows]
            ax.plot(x, y, marker="o", linewidth=2, color=color, alpha=0.9, label=mlabel)
        ax.axhline(0, color="#666666", linewidth=1)
        ax.set_title(f"Airbnb {corr_label}: Relative RMSE increase")
        ax.set_xlabel("Severity")
        ax.set_ylabel("RMSE increase vs clean (%)")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_robustness_heatmap(summary, output_path: Path):
    """Horizontal bar chart: % degradation from clean baseline for every model-corruption pair."""

    rows = []

    def _pct_deg_auroc(clean, worst):
        return 100.0 * abs(clean - worst) / clean if clean else 0.0

    def _pct_deg_rmse(clean, worst):
        return 100.0 * abs(worst - clean) / clean if clean else 0.0

    for corr in ["noise", "missingness", "imbalance"]:
        for model in ["RF", "XGB", "SVM-RBF"]:
            d = summary["adult"][corr][model]
            if corr == "imbalance":
                clean_auroc = d["worst_test_auroc"]
                worst_auroc = d["clean_test_auroc"]
            else:
                clean_auroc = d["clean_test_auroc"]
                worst_auroc = d["worst_test_auroc"]
            pct = _pct_deg_auroc(clean_auroc, worst_auroc)
            rows.append((f"Adult {corr.capitalize()}", model, pct))

    for model_key in ["Linear SVM", "Random Forest", "XGBoost"]:
        d = summary["imdb_token_dropout"][model_key]
        pct = _pct_deg_auroc(d["clean_test_auroc"], d["worst_test_auroc"])
        rows.append(("IMDB Token Dropout", model_key, pct))

    for model_key in ["Linear SVM", "Random Forest", "XGBoost"]:
        d = summary["imdb_domain_shift"][model_key]
        pct = _pct_deg_auroc(d["imdb_auroc"], d["amazon_auroc"])
        rows.append(("IMDB Domain Shift", model_key, pct))

    airbnb_model_keys = {
        "Random Forest": "rf",
        "Linear (Ridge fallback)": "linear",
        "XGBoost": "xgb",
    }
    AIRBNB_COMPARE_SEV = {"noise": 0.5, "missingness": 0.5}
    for corr in ["noise", "missingness"]:
        for model_label, model_file_key in airbnb_model_keys.items():
            stab_path = Path("outputs") / "week8_20260305" / f"airbnb_{corr}_{model_file_key}" / "stability_summary.json"
            stab_rows = load_json(stab_path)
            clean_row = [r for r in stab_rows if r["severity"] == 0.0][0]
            target_sev = AIRBNB_COMPARE_SEV[corr]
            worst_row = min(stab_rows, key=lambda r: abs(r["severity"] - target_sev))
            pct = _pct_deg_rmse(clean_row["test_rmse_mean"], worst_row["test_rmse_mean"])
            rows.append((f"Airbnb {corr.capitalize()}", model_label, pct))

    labels = [f"{exp}  |  {model}" for exp, model, _ in rows]
    values = [v for _, _, v in rows]
    experiments = [exp for exp, _, _ in rows]

    unique_exps = []
    for e in experiments:
        if e not in unique_exps:
            unique_exps.append(e)
    exp_colors = {
        "Adult Noise": "#2E86AB",
        "Adult Missingness": "#2E86AB",
        "Adult Imbalance": "#2E86AB",
        "IMDB Token Dropout": "#6C5CE7",
        "IMDB Domain Shift": "#6C5CE7",
        "Airbnb Noise": "#E94F37",
        "Airbnb Missingness": "#E94F37",
    }
    bar_colors = [exp_colors.get(e, "#888888") for e in experiments]

    CAP = 30
    clipped_values = [min(v, CAP) for v in values]

    fig, ax = plt.subplots(figsize=(11, 9))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, clipped_values, color=bar_colors, alpha=0.85, height=0.7, edgecolor="white", linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, values)):
        if val > CAP:
            ax.text(CAP - 0.3, i, f"{val:.0f}% -->", va="center", ha="right",
                    fontsize=9, fontweight="bold", color="white")
            bar.set_hatch("//")
        elif val < 0.5:
            ax.text(1.0, i, f"<0.5%", va="center", ha="left",
                    fontsize=8.5, fontweight="bold", color="#333333")
        else:
            ax.text(val + 0.4, i, f"{val:.1f}%", va="center", ha="left",
                    fontsize=9, fontweight="bold", color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Performance Degradation from Clean Baseline (%)", fontsize=11)
    ax.set_title("Robustness Overview: How Much Does Each Model Degrade?", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlim([0, CAP + 2])
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.grid(True, axis="x", alpha=0.25)

    ax.axvspan(0, 5, alpha=0.06, color="#27AE60")
    ax.axvspan(5, 15, alpha=0.06, color="#F39C12")
    ax.axvspan(15, CAP + 2, alpha=0.06, color="#E74C3C")
    ax.text(2.5, -1.0, "Robust", ha="center", fontsize=8, color="#27AE60", fontweight="bold")
    ax.text(10, -1.0, "Moderate", ha="center", fontsize=8, color="#F39C12", fontweight="bold")
    ax.text(22.5, -1.0, "Fragile", ha="center", fontsize=8, color="#E74C3C", fontweight="bold")

    sep_positions = []
    for i in range(1, len(experiments)):
        if experiments[i] != experiments[i - 1]:
            sep_positions.append(i - 0.5)
    for sp in sep_positions:
        ax.axhline(y=sp, color="#CCCCCC", linewidth=0.8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E86AB", alpha=0.85, label="Adult (AUROC-based)"),
        Patch(facecolor="#6C5CE7", alpha=0.85, label="IMDB (AUROC-based)"),
        Patch(facecolor="#E94F37", alpha=0.85, label="Airbnb (RMSE-based)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9, fontsize=9)

    ax.text(0.5, -0.06,
            "Each bar shows how much the model's primary metric (AUROC or RMSE) worsened\n"
            "at worst corruption vs clean data.  Hatched bars (//): value exceeds 30%, labeled inside bar.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#555555")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_adult_multimetric(outputs_root: Path, output_path: Path):
    """Adult noise: Accuracy vs F1 vs AUROC side-by-side to show metric sensitivity."""
    model_map = {
        "rf": ("Random Forest", "#2E86AB"),
        "xgb": ("XGBoost", "#E94F37"),
        "svm": ("SVM-RBF", "#44AF69"),
    }
    metric_cfg = [
        ("test_accuracy", "Accuracy"),
        ("test_f1", "F1 Score"),
        ("test_auroc", "AUROC"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (metric_key, metric_label) in zip(axes, metric_cfg):
        for key, (label, color) in model_map.items():
            rows = load_json(outputs_root / f"week6_rerun/adult_noise_{key}/stability_summary.json")
            rows = sorted(rows, key=lambda r: r["severity"])
            x = [r["severity"] for r in rows]
            y = [r[f"{metric_key}_mean"] for r in rows]
            yerr = [r.get(f"{metric_key}_std", 0.0) for r in rows]
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=2, color=color, alpha=0.9, label=label)
        ax.set_title(metric_label)
        ax.set_xlabel("Noise Severity")
        ax.set_ylabel(metric_label)
        ax.set_xlim([-0.02, 1.02])
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="best", framealpha=0.9)
    axes[0].set_ylim([0.83, 0.86])
    axes[1].set_ylim([0.76, 0.845])
    axes[2].set_ylim([0.65, 0.90])
    fig.suptitle("Adult Noise: Why Metric Choice Matters", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_imdb_domain_shift_sidebyside(summary, output_path: Path):
    """IMDB vs Amazon: side-by-side absolute performance for each metric."""
    models = ["Linear SVM", "Random Forest", "XGBoost"]
    metric_pairs = [
        ("imdb_accuracy", "amazon_accuracy", "Accuracy"),
        ("imdb_f1", "amazon_f1", "F1 Score"),
        ("imdb_auroc", "amazon_auroc", "AUROC"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(models))
    width = 0.32
    for ax, (imdb_key, amazon_key, label) in zip(axes, metric_pairs):
        imdb_vals = [summary["imdb_domain_shift"][m][imdb_key] for m in models]
        amazon_vals = [summary["imdb_domain_shift"][m][amazon_key] for m in models]
        ax.bar(x - width / 2, imdb_vals, width=width, color="#2E86AB", alpha=0.9, label="IMDB (in-domain)")
        ax.bar(x + width / 2, amazon_vals, width=width, color="#E94F37", alpha=0.9, label="Amazon (out-of-domain)")
        for i, (vi, va) in enumerate(zip(imdb_vals, amazon_vals)):
            drop = (vi - va) * 100
            ax.annotate(f"-{drop:.1f} pp", xy=(x[i], min(vi, va) - 0.01),
                        fontsize=8, fontweight="bold", color="#333333", ha="center", va="top")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=10, fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="upper right", framealpha=0.9)
    fig.suptitle("IMDB vs Amazon: In-Domain vs Out-of-Domain Performance", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_master_tables(summary, tables_dir: Path):
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: coverage
    rows = []
    for key, item in summary["coverage"].items():
        rows.append({
            "block_key": key,
            "completed_runs": item["got"],
            "expected_runs": item["expected"],
            "delta": item["got"] - item["expected"],
        })
    save_csv(
        tables_dir / "table01_run_coverage.csv",
        rows,
        ["block_key", "completed_runs", "expected_runs", "delta"],
    )

    # Table 2: Adult clean-vs-worst
    rows = []
    for corr, by_model in summary["adult"].items():
        for model, v in by_model.items():
            rows.append({
                "corruption": corr,
                "model": model,
                "clean_severity": v["clean_severity"],
                "worst_severity": v["worst_severity"],
                "clean_test_accuracy": v["clean_test_accuracy"],
                "worst_test_accuracy": v["worst_test_accuracy"],
                "delta_test_accuracy": v["delta_test_accuracy"],
                "clean_test_f1": v["clean_test_f1"],
                "worst_test_f1": v["worst_test_f1"],
                "delta_test_f1": v["delta_test_f1"],
                "clean_test_auroc": v["clean_test_auroc"],
                "worst_test_auroc": v["worst_test_auroc"],
                "delta_test_auroc": v["delta_test_auroc"],
            })
    save_csv(
        tables_dir / "table02_adult_clean_vs_worst.csv",
        rows,
        [
            "corruption", "model", "clean_severity", "worst_severity",
            "clean_test_accuracy", "worst_test_accuracy", "delta_test_accuracy",
            "clean_test_f1", "worst_test_f1", "delta_test_f1",
            "clean_test_auroc", "worst_test_auroc", "delta_test_auroc",
        ],
    )

    # Table 3: IMDB token dropout
    rows = []
    for model, v in summary["imdb_token_dropout"].items():
        rows.append({
            "model": model,
            "clean_severity": v["clean_severity"],
            "worst_severity": v["worst_severity"],
            "clean_test_accuracy": v["clean_test_accuracy"],
            "worst_test_accuracy": v["worst_test_accuracy"],
            "delta_test_accuracy": v["delta_test_accuracy"],
            "clean_test_f1": v["clean_test_f1"],
            "worst_test_f1": v["worst_test_f1"],
            "delta_test_f1": v["delta_test_f1"],
            "clean_test_auroc": v["clean_test_auroc"],
            "worst_test_auroc": v["worst_test_auroc"],
            "delta_test_auroc": v["delta_test_auroc"],
        })
    save_csv(
        tables_dir / "table03_imdb_token_dropout_clean_vs_worst.csv",
        rows,
        [
            "model", "clean_severity", "worst_severity",
            "clean_test_accuracy", "worst_test_accuracy", "delta_test_accuracy",
            "clean_test_f1", "worst_test_f1", "delta_test_f1",
            "clean_test_auroc", "worst_test_auroc", "delta_test_auroc",
        ],
    )

    # Table 4: IMDB domain shift
    rows = []
    for model, v in summary["imdb_domain_shift"].items():
        rows.append({"model": model, **v})
    save_csv(
        tables_dir / "table04_imdb_domain_shift_drops.csv",
        rows,
        [
            "model",
            "imdb_accuracy", "amazon_accuracy", "drop_accuracy_pp",
            "imdb_f1", "amazon_f1", "drop_f1_pp",
            "imdb_auroc", "amazon_auroc", "drop_auroc_pp",
        ],
    )

    # Table 5: Airbnb clean-vs-worst
    rows = []
    for corr, by_model in summary["airbnb"].items():
        for model, v in by_model.items():
            rows.append({
                "corruption": corr,
                "model": model,
                "clean_severity": v["clean_severity"],
                "worst_severity": v["worst_severity"],
                "clean_test_rmse": v["clean_test_rmse"],
                "worst_test_rmse": v["worst_test_rmse"],
                "delta_test_rmse": v["delta_test_rmse"],
                "clean_test_mae": v["clean_test_mae"],
                "worst_test_mae": v["worst_test_mae"],
                "delta_test_mae": v["delta_test_mae"],
            })
    save_csv(
        tables_dir / "table05_airbnb_clean_vs_worst.csv",
        rows,
        [
            "corruption", "model", "clean_severity", "worst_severity",
            "clean_test_rmse", "worst_test_rmse", "delta_test_rmse",
            "clean_test_mae", "worst_test_mae", "delta_test_mae",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Build thesis-ready master figure/table pack")
    parser.add_argument(
        "--outputs-root",
        type=str,
        default="outputs",
        help="Path to outputs root",
    )
    parser.add_argument(
        "--master-dir",
        type=str,
        default="outputs/master_20260309",
        help="Path to master output directory",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    master_dir = Path(args.master_dir)
    master_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = master_dir / "figures"
    tables_dir = master_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(master_dir / "master_summary_stats.json")

    plot_robustness_heatmap(summary, figures_dir / "fig01_robustness_heatmap.png")
    plot_adult_multimetric(outputs_root, figures_dir / "fig02_adult_metric_sensitivity.png")
    plot_adult_auroc_curves(outputs_root, figures_dir / "fig03_adult_auroc_curves.png")
    plot_imdb_token_dropout(outputs_root, figures_dir / "fig04_imdb_token_dropout_metrics.png")
    plot_imdb_domain_shift_sidebyside(summary, figures_dir / "fig05_imdb_domain_shift_sidebyside.png")
    plot_imdb_domain_shift(summary, figures_dir / "fig06_imdb_domain_shift_drops.png")
    plot_airbnb_absolute(outputs_root, figures_dir / "fig07_airbnb_absolute_curves.png")
    plot_airbnb_relative(outputs_root, figures_dir / "fig08_airbnb_relative_rmse_increase.png")

    export_master_tables(summary, tables_dir)

    print(f"Saved figures to: {figures_dir}")
    print(f"Saved tables to: {tables_dir}")


if __name__ == "__main__":
    main()

