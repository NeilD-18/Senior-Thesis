"""Run experiments across a grid of corruption severities."""
import argparse
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Import datasets and models to trigger registration
from .. import datasets, models, corruptions
from ..pipelines.corruption import run_corruption_experiment


def generate_severity_grid(min_severity=0.0, max_severity=1.0, n_points=11):
    """Generate evenly spaced severity values."""
    return np.linspace(min_severity, max_severity, n_points).tolist()


def _aggregate_stability(results, severities):
    """Compute mean and std per severity across seeds (for classification metrics)."""
    from collections import defaultdict
    by_severity = defaultdict(list)
    for r in results:
        by_severity[r['severity']].append({
            'val_metrics': r['val_metrics'],
            'test_metrics': r['test_metrics'],
        })
    out = []
    for sev in sorted(by_severity.keys()):
        runs = by_severity[sev]
        agg = {'severity': sev, 'n_seeds': len(runs)}
        for prefix in ('val_', 'test_'):
            for key in ('accuracy', 'f1', 'auroc'):
                k = prefix + key
                vals = []
                for run in runs:
                    m = run['val_metrics'] if prefix == 'val_' else run['test_metrics']
                    if key in m and m[key] is not None:
                        v = m[key]
                        vals.append(float(v) if hasattr(v, '__float__') else v)
                if vals:
                    agg[f'{k}_mean'] = float(np.mean(vals))
                    agg[f'{k}_std'] = float(np.std(vals))
        out.append(agg)
    return out


def _save_stability_json(path, stability):
    """Save stability summary as JSON (float-safe)."""
    with open(path, 'w') as f:
        json.dump(stability, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Run corruption experiments across severity grid'
    )
    parser.add_argument('--config', type=str, required=True, help='Path to base config YAML file')
    parser.add_argument('--min-severity', type=float, default=0.0, help='Minimum severity')
    parser.add_argument('--max-severity', type=float, default=1.0, help='Maximum severity')
    parser.add_argument('--n-points', type=int, default=11, help='Number of severity points')
    parser.add_argument('--severities', type=str, default=None,
                        help='Comma-separated list of severity values (overrides grid)')
    parser.add_argument('--output-dir', type=str, default='outputs/severity_grids',
                        help='Output directory for results')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds for stability (e.g. 42,43,44). Default: single seed from config.')
    parser.add_argument('--model', type=str, default=None,
                        help='Override model from config (e.g. xgboost, random_forest)')
    
    args = parser.parse_args()
    
    # Load base config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seeds = [base_config.get('seed', 42)]
    
    if args.model:
        base_config['model'] = args.model
        # Clear model_params when overriding model, since they may be incompatible
        base_config.pop('model_params', None)
    
    if args.severities:
        severities = [float(s.strip()) for s in args.severities.split(',')]
    else:
        severities = generate_severity_grid(args.min_severity, args.max_severity, args.n_points)
    
    n_runs = len(severities) * len(seeds)
    print(f"Running {n_runs} experiments: {len(severities)} severities x {len(seeds)} seeds")
    print(f"Severities: {severities}, Seeds: {seeds}, Model: {base_config['model']}")
    
    if 'corruption' not in base_config:
        raise ValueError("Config must include 'corruption' section")
    
    results = []
    for severity in tqdm(severities, desc="Severity grid"):
        for seed in seeds:
            config = base_config.copy()
            config['corruption'] = base_config['corruption'].copy()
            config['corruption']['severity'] = severity
            config['seed'] = seed
            config['output_dir'] = args.output_dir
            dataset_name = config['dataset']
            model_name = config['model']
            corruption_str = config['corruption'].get('type', 'none')
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"{dataset_name}_{model_name}_{corruption_str}_{severity:.2f}_seed{seed}_{ts}"
            try:
                result = run_corruption_experiment(config, run_name=run_name)
                result['severity'] = severity
                result['seed'] = seed
                results.append(result)
            except Exception as e:
                print(f"\nError at severity {severity} seed {seed}: {e}")
                continue
    
    summary_path = Path(args.output_dir) / 'severity_grid_summary.yaml'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'config_file': args.config,
        'severities': severities,
        'seeds': seeds,
        'model': base_config['model'],
        'results': [
            {
                'severity': r['severity'],
                'seed': r.get('seed'),
                'val_metrics': r['val_metrics'],
                'test_metrics': r['test_metrics'],
                'run_dir': str(r['run_dir'])
            }
            for r in results
        ]
    }
    
    if len(seeds) > 1:
        stability = _aggregate_stability(results, severities)
        stability_path = Path(args.output_dir) / 'stability_summary.json'
        _save_stability_json(stability_path, stability)
        summary['stability_aggregates'] = stability
        print(f"Stability summary saved to: {stability_path}")
    
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(x) for x in obj]
        return obj
    
    with open(summary_path, 'w') as f:
        yaml.dump(convert_numpy_types(summary), f, default_flow_style=False)
    
    print(f"Severity grid summary saved to: {summary_path}")
    print(f"Completed {len(results)}/{n_runs} experiments")


if __name__ == '__main__':
    main()
