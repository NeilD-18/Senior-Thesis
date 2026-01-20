"""I/O utilities for loading and saving data."""
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np


def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_pickle(obj, path: Path):
    """Save object as pickle."""
    ensure_dir(path.parent)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    """Load pickle object."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(obj, path: Path):
    """Save object as JSON."""
    ensure_dir(path.parent)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path):
    """Load JSON object."""
    with open(path, 'r') as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: Path):
    """Save DataFrame as CSV."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def get_run_dir(base_dir: Path, run_name: str) -> Path:
    """Get directory for a specific run."""
    run_dir = base_dir / run_name
    ensure_dir(run_dir)
    return run_dir
