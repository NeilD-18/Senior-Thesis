"""CLI for summarizing experiment results."""
import argparse
import pandas as pd
from pathlib import Path
import json
from ..common.io import load_json, save_csv


def main():
    parser = argparse.ArgumentParser(description='Summarize experiment results')
    parser.add_argument('--runs-dir', type=str, default='outputs/runs', help='Directory containing run results')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}")
        return
    
    # Collect all results
    results = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        metrics_file = run_dir / 'final_metrics.json'
        config_file = run_dir / 'config.json'
        
        if not metrics_file.exists() or not config_file.exists():
            continue
        
        metrics = load_json(metrics_file)
        config = load_json(config_file)
        
        result = {
            'run_name': run_dir.name,
            'dataset': config.get('dataset'),
            'model': config.get('model'),
            **metrics
        }
        results.append(result)
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(df, output_path)
    
    print(f"Summary saved to: {output_path}")
    print(f"\nTotal runs: {len(results)}")
    print("\nSummary:")
    print(df.to_string())


if __name__ == '__main__':
    main()
