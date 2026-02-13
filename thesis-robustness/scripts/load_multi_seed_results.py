#!/usr/bin/env python3
"""Load multi-seed severity grid results and compute mean/std per severity."""
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np


def extract_severity_and_seed_from_dirname(dirname):
    """Parse severity and seed from run dir name: ..._0.30_seed42_20250204_..."""
    # severity: _(\d+\.?\d*)_seed
    sev_match = re.search(r'_(\d+\.?\d*)_seed(\d+)_', dirname)
    if sev_match:
        return float(sev_match.group(1)), int(sev_match.group(2))
    # fallback: severity only (no seed in name)
    sev_only = re.search(r'_(\d+\.?\d*)_\d{8}_', dirname)
    if sev_only:
        return float(sev_only.group(1)), None
    return None, None


def load_results_from_grid_dir(grid_dir):
    """
    Load all run directories in grid_dir; group by (severity, seed) and compute mean/std per severity.
    Returns list of dicts: {severity, test_accuracy_mean, test_accuracy_std, ...}.
    """
    grid_path = Path(grid_dir)
    if not grid_path.exists():
        return []
    
    # Option 1: use precomputed stability_summary.json if present
    stability_file = grid_path / 'stability_summary.json'
    if stability_file.exists():
        with open(stability_file, 'r') as f:
            return json.load(f)
    
    # Option 2: scan run dirs and aggregate
    by_severity = defaultdict(list)
    for run_dir in grid_path.iterdir():
        if not run_dir.is_dir():
            continue
        metrics_file = run_dir / 'final_metrics.json'
        if not metrics_file.exists():
            continue
        severity, seed = extract_severity_and_seed_from_dirname(run_dir.name)
        if severity is None:
            continue
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        by_severity[severity].append({'seed': seed, 'metrics': metrics})
    
    out = []
    for sev in sorted(by_severity.keys()):
        runs = by_severity[sev]
        agg = {'severity': sev, 'n_seeds': len(runs)}
        for key in ('test_accuracy', 'test_f1', 'test_auroc', 'val_accuracy', 'val_f1', 'val_auroc'):
            vals = []
            for run in runs:
                v = run['metrics'].get(key)
                if v is not None:
                    vals.append(float(v))
            if vals:
                agg[f'{key}_mean'] = float(np.mean(vals))
                agg[f'{key}_std'] = float(np.std(vals))
        out.append(agg)
    return out


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_multi_seed_results.py <grid_dir> [grid_dir2 ...]")
        sys.exit(1)
    for d in sys.argv[1:]:
        r = load_results_from_grid_dir(d)
        print(f"{d}: {len(r)} severity levels")
        if r:
            print(json.dumps(r[:2], indent=2))
