"""Analyze and visualize severity grid results."""
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import base64


def load_severity_grid_summary(summary_path):
    """Load severity grid summary YAML."""
    import yaml.constructor
    
    # Custom loader to handle numpy serialization
    class SafeLoader(yaml.SafeLoader):
        pass
    
    def construct_python_object(loader, node):
        """Handle python object serialization by extracting values."""
        # For numpy scalars, try to extract the value
        if 'numpy' in str(node.tag) or 'scalar' in str(node.tag):
            # Try to get the value from the node
            if hasattr(node, 'value'):
                return float(node.value) if '.' in str(node.value) else int(node.value)
            # If it's a sequence, try to get the binary data
            if isinstance(node.value, list) and len(node.value) >= 2:
                try:
                    import base64
                    binary_data = base64.b64decode(node.value[-1])
                    return np.frombuffer(binary_data, dtype=np.float64)[0]
                except:
                    pass
        return None
    
    # Register custom constructor
    SafeLoader.add_constructor('tag:yaml.org,2002:python/object/apply', construct_python_object)
    
    try:
        with open(summary_path, 'r') as f:
            summary = yaml.load(f, Loader=SafeLoader)
    except:
        # Fallback: use safe_load and handle errors
        with open(summary_path, 'r') as f:
            summary = yaml.safe_load(f)
    
    return summary


def extract_numpy_value(value):
    """Extract value from numpy scalar serialization in YAML."""
    if isinstance(value, (int, float)):
        return float(value)
    
    # Handle numpy scalar serialization
    if isinstance(value, dict):
        # Check for binary serialization
        if '!!binary' in str(value) or '!!python/object/apply' in str(value):
            try:
                # Try to extract from binary data
                for k, v in value.items():
                    if isinstance(v, (bytes, str)):
                        if isinstance(v, str):
                            # Decode base64
                            binary_data = base64.b64decode(v)
                            return np.frombuffer(binary_data, dtype=np.float64)[0]
            except:
                pass
    
    # Fallback: try to convert string representation
    try:
        return float(str(value))
    except:
        return None


def extract_metrics(summary, metric='accuracy', split='test'):
    """Extract metrics across severities."""
    severities = []
    metrics = []
    
    for result in summary['results']:
        severity = result['severity']
        metric_value = result[f'{split}_metrics'].get(metric)
        
        if metric_value is not None:
            value = extract_numpy_value(metric_value)
            if value is not None:
                severities.append(severity)
                metrics.append(value)
    
    return np.array(severities), np.array(metrics)


def compute_robustness_summary(severities, metrics):
    """Compute robustness summary statistics."""
    if len(metrics) == 0:
        return None
    
    # Area under curve (normalized)
    if len(severities) > 1:
        auc = np.trapz(metrics, severities) / (severities.max() - severities.min())
    else:
        auc = metrics[0]
    
    # Performance drop
    clean_perf = metrics[0]  # severity = 0
    worst_perf = metrics.min()
    max_drop = clean_perf - worst_perf
    relative_drop = max_drop / clean_perf if clean_perf > 0 else 0
    
    # Degradation rate (slope)
    if len(severities) > 1:
        slope = np.polyfit(severities, metrics, 1)[0]  # Linear fit slope
    else:
        slope = 0
    
    return {
        'auc': auc,
        'clean_performance': clean_perf,
        'worst_performance': worst_perf,
        'max_drop': max_drop,
        'relative_drop': relative_drop,
        'degradation_slope': slope
    }


def plot_degradation_curves(summary, output_path=None):
    """Plot degradation curves for all metrics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = ['accuracy', 'f1', 'auroc']
    splits = ['test', 'val']
    colors = {'test': 'blue', 'val': 'orange'}
    markers = {'test': 'o', 'val': 's'}
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        for split in splits:
            try:
                severities, values = extract_metrics(summary, metric=metric, split=split)
                
                if len(severities) == 0:
                    continue
                
                # Sort by severity
                sort_idx = np.argsort(severities)
                severities = severities[sort_idx]
                values = values[sort_idx]
                
                ax.plot(severities, values, marker=markers[split], label=f'{split.upper()}', 
                       color=colors[split], linewidth=2, markersize=6, alpha=0.7)
            except Exception as e:
                print(f"Warning: Could not plot {split} {metric}: {e}")
        
        ax.set_xlabel('Corruption Severity', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} vs Severity', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([-0.05, 1.05])
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary_table(summary):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("SEVERITY GRID RESULTS SUMMARY")
    print("="*80)
    
    # Extract test metrics
    severities = []
    accuracies = []
    f1_scores = []
    aurocs = []
    
    for result in summary['results']:
        severity = result['severity']
        test_metrics = result['test_metrics']
        
        severities.append(severity)
        accuracies.append(extract_numpy_value(test_metrics.get('accuracy')))
        f1_scores.append(extract_numpy_value(test_metrics.get('f1')))
        aurocs.append(extract_numpy_value(test_metrics.get('auroc')))
    
    # Sort by severity
    sort_idx = np.argsort(severities)
    severities = np.array(severities)[sort_idx]
    accuracies = np.array([a for a in accuracies if a is not None])[sort_idx[:len(accuracies)]]
    f1_scores = np.array([f for f in f1_scores if f is not None])[sort_idx[:len(f1_scores)]]
    aurocs = np.array([a for a in aurocs if a is not None])[sort_idx[:len(aurocs)]]
    
    # Print table
    print(f"\n{'Severity':<10} {'Accuracy':<12} {'F1-Score':<12} {'AUROC':<12}")
    print("-" * 50)
    for i in range(len(severities)):
        acc_str = f"{accuracies[i]:.4f}" if i < len(accuracies) else "N/A"
        f1_str = f"{f1_scores[i]:.4f}" if i < len(f1_scores) else "N/A"
        auroc_str = f"{aurocs[i]:.4f}" if i < len(aurocs) else "N/A"
        print(f"{severities[i]:<10.2f} {acc_str:<12} {f1_str:<12} {auroc_str:<12}")
    
    # Compute robustness summaries
    print("\n" + "="*80)
    print("ROBUSTNESS SUMMARY STATISTICS")
    print("="*80)
    
    for metric_name, metric_values in [('Accuracy', accuracies), ('F1-Score', f1_scores), ('AUROC', aurocs)]:
        if len(metric_values) > 0 and metric_values[0] is not None:
            stats = compute_robustness_summary(severities[:len(metric_values)], metric_values)
            if stats:
                print(f"\n{metric_name}:")
                print(f"  Clean performance (severity=0): {stats['clean_performance']:.4f}")
                print(f"  Worst performance: {stats['worst_performance']:.4f}")
                print(f"  Maximum drop: {stats['max_drop']:.4f} ({stats['relative_drop']*100:.2f}%)")
                print(f"  Average performance (AUC): {stats['auc']:.4f}")
                print(f"  Degradation slope: {stats['degradation_slope']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze severity grid results')
    parser.add_argument('--summary', type=str, 
                       default='outputs/severity_grids/severity_grid_summary.yaml',
                       help='Path to severity grid summary YAML')
    parser.add_argument('--plot', type=str, default=None,
                       help='Path to save plot (default: show interactively)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting, only print summary')
    
    args = parser.parse_args()
    
    # Load summary
    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"Error: Summary file not found: {summary_path}")
        return
    
    print(f"Loading summary from: {summary_path}")
    summary = load_severity_grid_summary(summary_path)
    
    # Print summary table
    print_summary_table(summary)
    
    # Plot degradation curves
    if not args.no_plot:
        plot_path = args.plot or (summary_path.parent / 'degradation_curves.png')
        plot_degradation_curves(summary, output_path=plot_path)
    
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("""
Key observations:
1. Clean performance (severity=0): Baseline performance without corruption
2. Performance degradation: How quickly performance drops as severity increases
3. Relative drop: Percentage decrease from clean to worst performance
4. Degradation slope: Rate of performance decline (more negative = faster degradation)
5. AUC: Average performance across all severity levels (higher = more robust)

A robust model should:
- Maintain high performance at low severities
- Show gradual degradation (less negative slope)
- Have higher AUC (maintains performance across severity range)

See docs/RESULTS_GUIDE.md for detailed interpretation.
    """)


if __name__ == '__main__':
    main()
