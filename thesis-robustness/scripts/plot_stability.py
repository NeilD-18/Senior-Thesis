#!/usr/bin/env python3
"""Plot stability/variance (degradation curves with error bars) and model comparison."""
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def _extract_severity_seed(dirname):
    m = re.search(r'_(\d+\.?\d*)_seed(\d+)_', dirname)
    if m:
        return float(m.group(1)), int(m.group(2))
    m = re.search(r'_(\d+\.?\d*)_\d{8}_', dirname)
    if m:
        return float(m.group(1)), None
    return None, None


def load_results_from_grid_dir(grid_dir):
    """Load grid dir: use stability_summary.json if present, else aggregate from run dirs."""
    grid_path = Path(grid_dir)
    if not grid_path.exists():
        return []
    stability_file = grid_path / 'stability_summary.json'
    if stability_file.exists():
        with open(stability_file, 'r') as f:
            return json.load(f)
    by_severity = defaultdict(list)
    for run_dir in grid_path.iterdir():
        if not run_dir.is_dir():
            continue
        mfile = run_dir / 'final_metrics.json'
        if not mfile.exists():
            continue
        sev, seed = _extract_severity_seed(run_dir.name)
        if sev is None:
            continue
        with open(mfile, 'r') as f:
            metrics = json.load(f)
        by_severity[sev].append(metrics)
    out = []
    for sev in sorted(by_severity.keys()):
        runs = by_severity[sev]
        agg = {'severity': sev, 'n_seeds': len(runs)}
        for key in ('test_accuracy', 'test_f1', 'test_auroc', 'val_accuracy', 'val_f1', 'val_auroc'):
            vals = [float(r[key]) for r in runs if r.get(key) is not None]
            if vals:
                agg[f'{key}_mean'] = float(np.mean(vals))
                agg[f'{key}_std'] = float(np.std(vals))
        out.append(agg)
    return out

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colors and markers for multiple models
MODEL_STYLE = {
    'random_forest': {'color': '#2E86AB', 'marker': 'o', 'label': 'Random Forest'},
    'xgboost': {'color': '#E94F37', 'marker': 's', 'label': 'XGBoost'},
    'linear_svm': {'color': '#44AF69', 'marker': '^', 'label': 'Linear SVM'},
}
DEFAULT_STYLE = {'color': '#6C5CE7', 'marker': 'D', 'label': 'Model'}


def get_style(model_name, index):
    """Get plot style for a model (by name or index)."""
    if model_name and model_name in MODEL_STYLE:
        return MODEL_STYLE[model_name]
    colors = list(MODEL_STYLE.values())
    return colors[index % len(colors)] if index is not None else DEFAULT_STYLE


def plot_single_grid_with_error_bars(aggregates, output_path, title="Degradation (mean ± std across seeds)"):
    """Plot test metrics with error bars from stability aggregates."""
    if not aggregates:
        return
    severities = [a['severity'] for a in aggregates]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [('test_accuracy', 'Accuracy'), ('test_f1', 'F1-Score'), ('test_auroc', 'AUROC')]
    for idx, (key, label) in enumerate(metrics):
        ax = axes[idx]
        mean_key = f'{key}_mean'
        std_key = f'{key}_std'
        means = [a.get(mean_key) for a in aggregates if a.get(mean_key) is not None]
        stds = [a.get(std_key) for a in aggregates if a.get(std_key) is not None]
        sev_used = [s for s, a in zip(severities, aggregates) if a.get(mean_key) is not None]
        if not means:
            continue
        ax.errorbar(sev_used, means, yerr=stds, marker='o', capsize=4, capthick=1.5,
                    linewidth=2, color='#2E86AB', label='Test (mean ± std)')
        ax.set_xlabel('Corruption Severity')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9)
        ax.set_xlim([-0.02, max(severities) + 0.02])
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_model_comparison(grid_dirs_with_labels, output_path, title="Model comparison (test accuracy)"):
    """
    grid_dirs_with_labels: list of (grid_dir_path, model_label) e.g. [("outputs/week5/adult_noise_rf", "RF"), ...]
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [('test_accuracy', 'Accuracy'), ('test_f1', 'F1-Score'), ('test_auroc', 'AUROC')]
    for idx, (key, ylabel) in enumerate(metrics):
        ax = axes[idx]
        mean_key = f'{key}_mean'
        std_key = f'{key}_std'
        for i, (grid_dir, label) in enumerate(grid_dirs_with_labels):
            agg = load_results_from_grid_dir(grid_dir)
            if not agg:
                continue
            severities = [a['severity'] for a in agg]
            means = [a.get(mean_key) for a in agg]
            stds = [a.get(std_key) for a in agg]
            if not any(m is not None for m in means):
                continue
            style = get_style(label.lower().replace(' ', '_'), i)
            ax.errorbar(severities, means, yerr=stds, marker=style['marker'], capsize=3, capthick=1.2,
                        linewidth=2, color=style['color'], label=label, alpha=0.9)
        ax.set_xlabel('Corruption Severity')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9)
        ax.set_xlim([-0.02, 1.02])
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot stability/variance and model comparison')
    parser.add_argument('--grid-dir', type=str, action='append', default=[],
                        help='Severity grid directory (can repeat for model comparison)')
    parser.add_argument('--label', type=str, action='append', default=[],
                        help='Label for each grid (e.g. RF, XGB); same order as --grid-dir')
    parser.add_argument('--output', type=str, default='outputs/week5/stability_curves.png',
                        help='Output plot path')
    parser.add_argument('--title', type=str, default='Week 5: Stability and model comparison')
    args = parser.parse_args()
    
    if not args.grid_dir:
        print("Provide at least one --grid-dir")
        return
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    if len(args.grid_dir) == 1:
        # Single grid: plot with error bars
        agg = load_results_from_grid_dir(args.grid_dir[0])
        if not agg:
            print(f"No results in {args.grid_dir[0]}")
            return
        plot_single_grid_with_error_bars(agg, args.output, title=args.title)
    else:
        # Multiple grids: model comparison
        labels = args.label if len(args.label) == len(args.grid_dir) else [Path(d).name for d in args.grid_dir]
        pairs = list(zip(args.grid_dir, labels))
        plot_model_comparison(pairs, args.output, title=args.title)


if __name__ == '__main__':
    main()
