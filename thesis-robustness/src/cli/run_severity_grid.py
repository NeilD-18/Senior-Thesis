"""Run experiments across a grid of corruption severities."""
import argparse
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import datasets and models to trigger registration
from .. import datasets, models, corruptions
from ..pipelines.corruption import run_corruption_experiment


def generate_severity_grid(min_severity=0.0, max_severity=1.0, n_points=11):
    """Generate evenly spaced severity values."""
    return np.linspace(min_severity, max_severity, n_points).tolist()


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
    
    args = parser.parse_args()
    
    # Load base config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Determine severity values
    if args.severities:
        severities = [float(s.strip()) for s in args.severities.split(',')]
    else:
        severities = generate_severity_grid(args.min_severity, args.max_severity, args.n_points)
    
    print(f"Running experiments for {len(severities)} severity values: {severities}")
    
    # Ensure corruption config exists
    if 'corruption' not in base_config:
        raise ValueError("Config must include 'corruption' section")
    
    # Run experiments
    results = []
    for severity in tqdm(severities, desc="Severity grid"):
        # Create config for this severity
        config = base_config.copy()
        config['corruption'] = base_config['corruption'].copy()
        config['corruption']['severity'] = severity
        
        # Update output directory
        config['output_dir'] = args.output_dir
        
        # Run experiment
        try:
            result = run_corruption_experiment(config)
            result['severity'] = severity
            results.append(result)
        except Exception as e:
            print(f"\nError at severity {severity}: {e}")
            continue
    
    # Save summary
    summary_path = Path(args.output_dir) / 'severity_grid_summary.yaml'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'config_file': args.config,
        'severities': severities,
        'results': [
            {
                'severity': r['severity'],
                'val_metrics': r['val_metrics'],
                'test_metrics': r['test_metrics'],
                'run_dir': str(r['run_dir'])
            }
            for r in results
        ]
    }
    
    # Convert numpy types to native Python types for YAML serialization
    def convert_numpy_types(obj):
        """Recursively convert numpy types to native Python types."""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    summary_clean = convert_numpy_types(summary)
    
    with open(summary_path, 'w') as f:
        yaml.dump(summary_clean, f, default_flow_style=False)
    
    print(f"\nSeverity grid summary saved to: {summary_path}")
    print(f"Completed {len(results)}/{len(severities)} experiments")


if __name__ == '__main__':
    main()
