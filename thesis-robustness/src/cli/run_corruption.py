"""CLI entrypoint for corruption experiments."""
import argparse
import yaml
from pathlib import Path

# Import datasets and models to trigger registration
from .. import datasets, models, corruptions
from ..pipelines.corruption import run_corruption_experiment


def main():
    parser = argparse.ArgumentParser(description='Run corruption robustness experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--run-name', type=str, default=None, help='Optional run name')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run experiment
    results = run_corruption_experiment(config, run_name=args.run_name)
    
    print(f"\nResults saved to: {results['run_dir']}")


if __name__ == '__main__':
    main()
