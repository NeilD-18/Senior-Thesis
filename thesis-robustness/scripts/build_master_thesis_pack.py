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
    """Single-figure overview: every model x every corruption, color = degradation magnitude."""
    import matplotlib.colors as mcolors

    rows_cfg = [
        ("Adult Noise",        "adult",  "noise",       "RF",                       "delta_test_auroc", 100),
        ("Adult Noise",        "adult",  "noise",       "XGB",                      "delta_test_auroc", 100),
        ("Adult Noise",        "adult",  "noise",       "SVM-RBF",                  "delta_test_auroc", 100),
        ("Adult Missingness",  "adult",  "missingness", "RF",                       "delta_test_auroc", 100),
        ("Adult Missingness",  "adult",  "missingness", "XGB",                      "delta_test_auroc", 100),
        ("Adult Missingness",  "adult",  "missingness", "SVM-RBF",                  "delta_test_auroc", 100),
        ("Adult Imbalance",    "adult",  "imbalance",   "RF",                       "delta_test_auroc", 100),
        ("Adult Imbalance",    "adult",  "imbalance",   "XGB",                      "delta_test_auroc", 100),
        ("Adult Imbalance",    "adult",  "imbalance",   "SVM-RBF",                  "delta_test_auroc", 100),
        ("IMDB Dropout",       "imdb_token_dropout", None, "Linear SVM",            "delta_test_auroc", 100),
        ("IMDB Dropout",       "imdb_token_dropout", None, "Random Forest",         "delta_test_auroc", 100),
        ("IMDB Dropout",       "imdb_token_dropout", None, "XGBoost",               "delta_test_auroc", 100),
        ("IMDB Domain Shift",  "imdb_domain_shift",  None, "Linear SVM",            "drop_auroc_pp",   -1),
        ("IMDB Domain Shift",  "imdb_domain_shift",  None, "Random Forest",         "drop_auroc_pp",   -1),
        ("IMDB Domain Shift",  "imdb_domain_shift",  None, "XGBoost",               "drop_auroc_pp",   -1),
        ("Airbnb Noise",       "airbnb", "noise",       "Random Forest",            "delta_test_rmse",  1),
        ("Airbnb Noise",       "airbnb", "noise",       "Linear (Ridge fallback)",  "delta_test_rmse",  1),
        ("Airbnb Noise",       "airbnb", "noise",       "XGBoost",                  "delta_test_rmse",  1),
        ("Airbnb Missingness", "airbnb", "missingness", "Random Forest",            "delta_test_rmse",  1),
        ("Airbnb Missingness", "airbnb", "missingness", "Linear (Ridge fallback)",  "delta_test_rmse",  1),
        ("Airbnb Missingness", "airbnb", "missingness", "XGBoost",                  "delta_test_rmse",  1),
    ]

    experiment_labels = []
    model_labels = []
    raw_values = []
    display_values = []

    for exp_label, block, corr, model, metric_key, scale in rows_cfg:
        if corr is not None:
            val = summary[block][corr][model][metric_key]
        else:
            val = summary[block][model][metric_key]

        if block == "airbnb":
            display = val
            severity = -abs(val)
        elif metric_key == "drop_auroc_pp":
            display = -val
            severity = -val
        else:
            display = val * (scale if scale != 100 else 1)
            if scale == 100:
                display = val * 100
            severity = val * 100
        raw_values.append(severity)
        display_values.append(display)
        experiment_labels.append(exp_label)
        model_labels.append(model)

    unique_experiments = []
    for e in experiment_labels:
        if e not in unique_experiments:
            unique_experiments.append(e)
    unique_models_per_exp = {}
    for e, m in zip(experiment_labels, model_labels):
        unique_models_per_exp.setdefault(e, [])
        if m not in unique_models_per_exp[e]:
            unique_models_per_exp[e].append(m)

    n_rows = len(unique_experiments)
    max_models = max(len(v) for v in unique_models_per_exp.values())

    grid = np.full((n_rows, max_models), np.nan)
    display_grid = np.full((n_rows, max_models), np.nan)
    label_grid = [[""] * max_models for _ in range(n_rows)]
    is_rmse = [False] * n_rows

    idx = 0
    for i, exp in enumerate(unique_experiments):
        models = unique_models_per_exp[exp]
        if "airbnb" in exp.lower():
            is_rmse[i] = True
        for j, model in enumerate(models):
            grid[i, j] = raw_values[idx]
            display_grid[i, j] = display_values[idx]
            label_grid[i][j] = model
            idx += 1

    auroc_vals = [grid[i, j] for i in range(n_rows) for j in range(max_models) if not np.isnan(grid[i, j]) and not is_rmse[i]]
    rmse_vals = [grid[i, j] for i in range(n_rows) for j in range(max_models) if not np.isnan(grid[i, j]) and is_rmse[i]]

    vmin_auroc = min(auroc_vals) if auroc_vals else -15
    vmax_auroc = max(max(auroc_vals), 0) if auroc_vals else 5
    vmin_rmse = min(rmse_vals) if rmse_vals else -0.7
    vmax_rmse = 0

    norm_auroc = mcolors.TwoSlopeNorm(vmin=vmin_auroc, vcenter=0, vmax=max(vmax_auroc, 1))

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    cmap = plt.cm.RdYlGn

    for i in range(n_rows):
        for j in range(max_models):
            if np.isnan(grid[i, j]):
                continue
            val = grid[i, j]
            if is_rmse[i]:
                normed = 1.0 - min(abs(val) / 0.65, 1.0)
                color = cmap(normed * 0.5)
            else:
                color = cmap(norm_auroc(val))

            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="white", linewidth=2))

            dv = display_grid[i, j]
            if is_rmse[i]:
                cell_text = f"+{dv:.3f}"
            elif dv > 0:
                cell_text = f"+{dv:.1f} pp"
            else:
                cell_text = f"{dv:.1f} pp"
            text_color = "white" if abs(val) > (8 if not is_rmse[i] else 0.3) else "#222222"
            ax.text(j, i + 0.12, cell_text, ha="center", va="center", fontsize=9, fontweight="bold", color=text_color)
            ax.text(j, i - 0.18, label_grid[i][j], ha="center", va="center", fontsize=7.5, color=text_color, alpha=0.85)

    ax.set_xlim(-0.5, max_models - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(unique_experiments, fontsize=9.5)
    ax.set_title("Robustness Overview: Performance Change at Worst Severity", fontsize=13, fontweight="bold", pad=14)
    ax.text(0.5, -0.04, "Green = robust (small change)     Red = fragile (large degradation)\nClassification cells: AUROC change in pp     Regression cells: RMSE increase (absolute)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#555555")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

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

