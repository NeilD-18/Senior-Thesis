#!/usr/bin/env python3
"""Generate professional plots for all severity grid results."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Professional styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'test': '#2E86AB',      # Steel blue
    'validation': '#E94F37'  # Vermillion
}

MARKERS = {
    'test': 'o',
    'validation': 's'
}


def extract_severity_from_dirname(dirname):
    """Extract severity value from directory name."""
    # Pattern: ..._corruption_type_0.30_...
    match = re.search(r'_(\d+\.?\d*)_\d{8}_', dirname)
    if match:
        return float(match.group(1))
    return None


def load_results_from_grid(grid_dir):
    """Load results from a single severity grid directory."""
    grid_path = Path(grid_dir)
    results = []
    
    for run_dir in sorted(grid_path.iterdir()):
        if not run_dir.is_dir():
            continue
        
        metrics_file = run_dir / 'final_metrics.json'
        if not metrics_file.exists():
            continue
        
        severity = extract_severity_from_dirname(run_dir.name)
        if severity is None:
            continue
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        results.append({
            'severity': severity,
            'test_accuracy': metrics.get('test_accuracy'),
            'test_f1': metrics.get('test_f1'),
            'test_auroc': metrics.get('test_auroc'),
            'test_rmse': metrics.get('test_rmse'),
            'test_mae': metrics.get('test_mae'),
            'val_accuracy': metrics.get('val_accuracy'),
            'val_f1': metrics.get('val_f1'),
            'val_auroc': metrics.get('val_auroc'),
            'val_rmse': metrics.get('val_rmse'),
            'val_mae': metrics.get('val_mae'),
        })
    
    results.sort(key=lambda x: x['severity'])
    return results


def plot_classification_degradation(results, output_path, title="Degradation Curves"):
    """Plot classification metrics degradation."""
    if not results:
        return
    
    severities = [r['severity'] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    metrics = [
        ('accuracy', 'Accuracy'),
        ('f1', 'F1-Score'),
        ('auroc', 'AUROC')
    ]
    
    for idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[idx]
        
        # Test metrics
        test_values = [r[f'test_{metric_key}'] for r in results if r.get(f'test_{metric_key}') is not None]
        test_sev = [s for s, r in zip(severities, results) if r.get(f'test_{metric_key}') is not None]
        
        if test_values:
            ax.plot(test_sev, test_values, 
                   marker=MARKERS['test'], 
                   color=COLORS['test'], 
                   linewidth=2, 
                   markersize=8, 
                   label='Test',
                   zorder=3)
        
        # Validation metrics
        val_values = [r[f'val_{metric_key}'] for r in results if r.get(f'val_{metric_key}') is not None]
        val_sev = [s for s, r in zip(severities, results) if r.get(f'val_{metric_key}') is not None]
        
        if val_values:
            ax.plot(val_sev, val_values, 
                   marker=MARKERS['validation'], 
                   color=COLORS['validation'], 
                   linewidth=2, 
                   markersize=7,
                   linestyle='--',
                   label='Validation',
                   alpha=0.8,
                   zorder=2)
        
        ax.set_xlabel('Corruption Severity')
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc='best', framealpha=0.9)
        ax.set_xlim([-0.02, max(severities) + 0.02])
        
        if test_values:
            ymin = min(min(test_values), min(val_values)) - 0.02 if val_values else min(test_values) - 0.02
            ymax = max(max(test_values), max(val_values)) + 0.02 if val_values else max(test_values) + 0.02
            # Ensure reasonable y-axis range
            if ymax - ymin < 0.05:
                center = (ymax + ymin) / 2
                ymin = center - 0.05
                ymax = center + 0.05
            ax.set_ylim([ymin, ymax])
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_regression_degradation(results, output_path, title="Degradation Curves"):
    """Plot regression metrics degradation."""
    if not results:
        return
    
    severities = [r['severity'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    metrics = [
        ('rmse', 'RMSE'),
        ('mae', 'MAE')
    ]
    
    for idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[idx]
        
        test_values = [r[f'test_{metric_key}'] for r in results if r.get(f'test_{metric_key}') is not None]
        test_sev = [s for s, r in zip(severities, results) if r.get(f'test_{metric_key}') is not None]
        
        if test_values:
            ax.plot(test_sev, test_values, 
                   marker=MARKERS['test'], 
                   color=COLORS['test'], 
                   linewidth=2, 
                   markersize=8, 
                   label='Test',
                   zorder=3)
        
        val_values = [r[f'val_{metric_key}'] for r in results if r.get(f'val_{metric_key}') is not None]
        val_sev = [s for s, r in zip(severities, results) if r.get(f'val_{metric_key}') is not None]
        
        if val_values:
            ax.plot(val_sev, val_values, 
                   marker=MARKERS['validation'], 
                   color=COLORS['validation'], 
                   linewidth=2, 
                   markersize=7,
                   linestyle='--',
                   label='Validation',
                   alpha=0.8,
                   zorder=2)
        
        ax.set_xlabel('Corruption Severity')
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc='best', framealpha=0.9)
        ax.set_xlim([-0.02, max(severities) + 0.02])
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def print_results_table(results, name, is_regression=False):
    """Print formatted results table."""
    print(f"\n{'='*70}")
    print(f"{name.upper()}")
    print('='*70)
    
    if is_regression:
        print(f"{'Severity':<12} {'Test RMSE':<14} {'Test MAE':<14}")
        print('-' * 40)
        for r in results:
            rmse = r.get('test_rmse', 'N/A')
            mae = r.get('test_mae', 'N/A')
            if isinstance(rmse, (int, float)):
                rmse = f"{rmse:.4f}"
            if isinstance(mae, (int, float)):
                mae = f"{mae:.4f}"
            print(f"{r['severity']:<12.2f} {rmse:<14} {mae:<14}")
    else:
        print(f"{'Severity':<12} {'Test Acc':<14} {'Test F1':<14} {'Test AUROC':<14}")
        print('-' * 56)
        for r in results:
            acc = r.get('test_accuracy', 'N/A')
            f1 = r.get('test_f1', 'N/A')
            auroc = r.get('test_auroc', 'N/A')
            if isinstance(acc, (int, float)):
                acc = f"{acc:.4f}"
            if isinstance(f1, (int, float)):
                f1 = f"{f1:.4f}"
            if isinstance(auroc, (int, float)):
                auroc = f"{auroc:.4f}"
            print(f"{r['severity']:<12.2f} {acc:<14} {f1:<14} {auroc:<14}")


def compute_degradation_summary(results, metric_key='test_accuracy'):
    """Compute degradation summary statistics."""
    values = [r[metric_key] for r in results if r.get(metric_key) is not None]
    severities = [r['severity'] for r in results if r.get(metric_key) is not None]
    
    if len(values) < 2:
        return None
    
    clean_value = values[0] if severities[0] == 0.0 else values[0]
    worst_value = min(values)
    
    # Compute slope using linear regression
    coeffs = np.polyfit(severities, values, 1)
    slope = coeffs[0]
    
    # Area under degradation curve (normalized)
    auc = np.trapz(values, severities) / max(severities)
    
    return {
        'clean': clean_value,
        'worst': worst_value,
        'drop': clean_value - worst_value,
        'drop_pct': (clean_value - worst_value) / clean_value * 100 if clean_value > 0 else 0,
        'slope': slope,
        'auc': auc
    }


def main():
    base_dir = Path('outputs/severity_grids')
    output_dir = Path('outputs/summary')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING PLOTS AND SUMMARIES")
    print("=" * 70)
    
    # Process each severity grid
    grids = {
        'adult_noise': ('Adult Income - Additive Noise', False),
        'adult_missingness': ('Adult Income - Missingness', False),
        'imdb_token_dropout': ('IMDB - Token Dropout', False),
    }
    
    all_summaries = {}
    
    for grid_name, (display_name, is_regression) in grids.items():
        grid_path = base_dir / grid_name
        if not grid_path.exists():
            print(f"Skipping {grid_name} (not found)")
            continue
        
        results = load_results_from_grid(grid_path)
        if not results:
            print(f"Skipping {grid_name} (no results)")
            continue
        
        # Print table
        print_results_table(results, display_name, is_regression)
        
        # Generate plot
        plot_path = output_dir / f'{grid_name}_degradation.png'
        if is_regression:
            plot_regression_degradation(results, plot_path, title=display_name)
        else:
            plot_classification_degradation(results, plot_path, title=display_name)
        
        # Compute summary
        if not is_regression:
            summary = compute_degradation_summary(results, 'test_accuracy')
            if summary:
                all_summaries[grid_name] = summary
    
    # Print overall summary
    if all_summaries:
        print("\n" + "=" * 70)
        print("DEGRADATION SUMMARY")
        print("=" * 70)
        print(f"{'Experiment':<30} {'Clean':<10} {'Worst':<10} {'Drop %':<10} {'Slope':<10}")
        print("-" * 70)
        for name, s in all_summaries.items():
            print(f"{name:<30} {s['clean']:.4f}    {s['worst']:.4f}    {s['drop_pct']:.1f}%      {s['slope']:.4f}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
