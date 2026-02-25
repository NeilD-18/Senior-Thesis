#!/usr/bin/env python3
"""Create additional Week 7 summary plots."""
import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


MODEL_ORDER = ['linear_svm', 'xgboost', 'random_forest']
MODEL_LABEL = {
    'linear_svm': 'Linear SVM',
    'xgboost': 'XGBoost',
    'random_forest': 'Random Forest',
}
MODEL_COLOR = {
    'linear_svm': '#2E86AB',
    'xgboost': '#E94F37',
    'random_forest': '#44AF69',
}
METRICS = [('accuracy', 'Accuracy'), ('f1', 'F1-Score'), ('auroc', 'AUROC')]


def _load_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def load_token_dropout_summaries(week7_dir: Path):
    """Load stability_summary.json per model for IMDB token-dropout runs."""
    dir_map = {
        'linear_svm': week7_dir / 'imdb_token_dropout_linear_svm' / 'stability_summary.json',
        'xgboost': week7_dir / 'imdb_token_dropout_xgb' / 'stability_summary.json',
        'random_forest': week7_dir / 'imdb_token_dropout_rf' / 'stability_summary.json',
    }
    out = {}
    for model, path in dir_map.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing summary: {path}")
        rows = _load_json(path)
        out[model] = sorted(rows, key=lambda r: r['severity'])
    return out


def load_domain_shift_summary(week7_dir: Path):
    """Load aggregated IMDB->Amazon summary."""
    path = week7_dir / 'imdb_to_amazon' / 'domain_shift_summary.json'
    if not path.exists():
        raise FileNotFoundError(f"Missing summary: {path}")
    return _load_json(path)


def plot_token_dropout_delta(token_data, output_path: Path):
    """Plot delta from clean severity (mean ± std, in percentage points)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (metric, metric_label) in zip(axes, METRICS):
        for model in MODEL_ORDER:
            rows = token_data[model]
            baseline = rows[0][f'test_{metric}_mean']
            severities = [r['severity'] for r in rows]
            delta_pp = [(r[f'test_{metric}_mean'] - baseline) * 100.0 for r in rows]
            std_pp = [r.get(f'test_{metric}_std', 0.0) * 100.0 for r in rows]
            ax.errorbar(
                severities,
                delta_pp,
                yerr=std_pp,
                marker='o',
                linewidth=2,
                capsize=3,
                color=MODEL_COLOR[model],
                label=MODEL_LABEL[model],
                alpha=0.9,
            )
        ax.axhline(0.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
        ax.set_xlabel('Token Dropout Severity')
        ax.set_ylabel(f'{metric_label} delta from clean (pp)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 0.52])
    axes[0].legend(loc='best', framealpha=0.9)
    fig.suptitle('IMDB Token Dropout: Change from Clean Baseline (mean ± std)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_domain_shift_grouped(summary, output_path: Path):
    """Plot grouped bars: IMDB in-domain vs Amazon out-of-domain."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.36

    for ax, (metric, metric_label) in zip(axes, METRICS):
        imdb_vals = [summary[f'{m}_imdb_test'][f'{metric}_mean'] * 100.0 for m in MODEL_ORDER]
        amazon_vals = [summary[f'{m}_amazon'][f'{metric}_mean'] * 100.0 for m in MODEL_ORDER]
        imdb_std = [summary[f'{m}_imdb_test'].get(f'{metric}_std', 0.0) * 100.0 for m in MODEL_ORDER]
        amazon_std = [summary[f'{m}_amazon'].get(f'{metric}_std', 0.0) * 100.0 for m in MODEL_ORDER]

        ax.bar(x - width / 2, imdb_vals, width, yerr=imdb_std, capsize=3, color='#2E86AB', label='IMDB test')
        ax.bar(x + width / 2, amazon_vals, width, yerr=amazon_std, capsize=3, color='#E94F37', label='Amazon')
        ax.set_xticks(x, [MODEL_LABEL[m] for m in MODEL_ORDER], rotation=15)
        ax.set_ylabel(f'{metric_label} (%)')
        ax.grid(True, axis='y', alpha=0.3)
    axes[0].legend(loc='best', framealpha=0.9)
    fig.suptitle('IMDB -> Amazon Domain Shift: In-Domain vs Out-of-Domain', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_domain_shift_drop(summary, output_path: Path):
    """Plot performance drop (IMDB - Amazon) in percentage points per model/metric."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(MODEL_ORDER))
    colors = [MODEL_COLOR[m] for m in MODEL_ORDER]

    for ax, (metric, metric_label) in zip(axes, METRICS):
        drops = []
        for model in MODEL_ORDER:
            imdb = summary[f'{model}_imdb_test'][f'{metric}_mean']
            amazon = summary[f'{model}_amazon'][f'{metric}_mean']
            drops.append((imdb - amazon) * 100.0)
        ax.bar(x, drops, color=colors, alpha=0.9)
        ax.set_xticks(x, [MODEL_LABEL[m] for m in MODEL_ORDER], rotation=15)
        ax.set_ylabel(f'{metric_label} drop (pp)')
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle('IMDB -> Amazon Domain Shift: Performance Drop by Model', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Week 7 summary plots')
    parser.add_argument('--week7-dir', type=str, default='outputs/week7', help='Week 7 outputs directory')
    args = parser.parse_args()

    week7_dir = Path(args.week7_dir)
    token_data = load_token_dropout_summaries(week7_dir)
    summary = load_domain_shift_summary(week7_dir)

    plot_token_dropout_delta(token_data, week7_dir / 'imdb_token_dropout_delta_from_clean.png')
    plot_domain_shift_grouped(summary, week7_dir / 'imdb_to_amazon_grouped_metrics.png')
    plot_domain_shift_drop(summary, week7_dir / 'imdb_to_amazon_drop_pp.png')


if __name__ == '__main__':
    main()
