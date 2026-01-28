#!/usr/bin/env python3
"""Simple script to plot results from severity grid runs."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


def extract_severity_from_dirname(dirname):
    """Extract severity value from directory name."""
    # Pattern: ..._additive_noise_0.30_...
    match = re.search(r'_(\d+\.?\d*)_\d{8}_', dirname)
    if match:
        return float(match.group(1))
    return None


def load_results_from_dirs(base_dir):
    """Load results from individual run directories."""
    base_path = Path(base_dir)
    results = []
    
    for run_dir in sorted(base_path.iterdir()):
        if not run_dir.is_dir():
            continue
        
        metrics_file = run_dir / 'final_metrics.json'
        if not metrics_file.exists():
            continue
        
        # Extract severity from directory name
        severity = extract_severity_from_dirname(run_dir.name)
        if severity is None:
            continue
        
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        results.append({
            'severity': severity,
            'test_accuracy': metrics.get('test_accuracy'),
            'test_f1': metrics.get('test_f1'),
            'test_auroc': metrics.get('test_auroc'),
            'val_accuracy': metrics.get('val_accuracy'),
            'val_f1': metrics.get('val_f1'),
            'val_auroc': metrics.get('val_auroc'),
        })
    
    # Sort by severity
    results.sort(key=lambda x: x['severity'])
    return results


def plot_results(results, output_path):
    """Plot degradation curves."""
    severities = [r['severity'] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [
        ('accuracy', 'Accuracy'),
        ('f1', 'F1-Score'),
        ('auroc', 'AUROC')
    ]
    
    for idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[idx]
        
        # Test metrics
        test_values = [r[f'test_{metric_key}'] for r in results if r.get(f'test_{metric_key}') is not None]
        test_severities = [s for s, r in zip(severities, results) if r.get(f'test_{metric_key}') is not None]
        
        if test_values:
            ax.plot(test_severities, test_values, 'o-', label='Test', 
                   color='blue', linewidth=2, markersize=8, alpha=0.7)
        
        # Validation metrics
        val_values = [r[f'val_{metric_key}'] for r in results if r.get(f'val_{metric_key}') is not None]
        val_severities = [s for s, r in zip(severities, results) if r.get(f'val_{metric_key}') is not None]
        
        if val_values:
            ax.plot(val_severities, val_values, 's-', label='Validation', 
                   color='orange', linewidth=2, markersize=8, alpha=0.7)
        
        ax.set_xlabel('Corruption Severity', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} vs Severity', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def print_summary(results):
    """Print summary table."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Severity':<10} {'Test Acc':<12} {'Test F1':<12} {'Test AUROC':<12}")
    print("-" * 50)
    
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
        
        print(f"{r['severity']:<10.2f} {acc:<12} {f1:<12} {auroc:<12}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot severity grid results')
    parser.add_argument('--dir', type=str, 
                       default='outputs/severity_grids',
                       help='Directory containing severity grid runs')
    parser.add_argument('--plot', type=str,
                       default='outputs/severity_grids/degradation_curves.png',
                       help='Output path for plot')
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.dir}")
    results = load_results_from_dirs(args.dir)
    
    if not results:
        print(f"Error: No results found in {args.dir}")
        return
    
    print(f"Found {len(results)} experiments")
    
    print_summary(results)
    plot_results(results, args.plot)
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
