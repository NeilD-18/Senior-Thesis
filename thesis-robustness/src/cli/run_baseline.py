"""CLI entrypoint for baseline experiments."""
import argparse
import yaml
from pathlib import Path

# Import datasets and models to trigger registration
from .. import datasets, models
from ..pipelines.baseline import run_baseline


def main():
    parser = argparse.ArgumentParser(description='Run baseline experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--run-name', type=str, default=None, help='Optional run name')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run experiment
    results = run_baseline(config, run_name=args.run_name)
    
    print(f"\nResults saved to: {results['run_dir']}")


if __name__ == '__main__':
    main()
