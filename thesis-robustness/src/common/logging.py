"""Logging utilities for experiment tracking."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from .io import ensure_dir, save_json, save_csv


class RunLogger:
    """Logger for experiment runs."""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        ensure_dir(run_dir)
        self.metrics = []
        
    def log_metrics(self, metrics: Dict[str, Any], step: int = None):
        """Log metrics for a step."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            **metrics
        }
        self.metrics.append(entry)
        
    def log_config(self, config: Dict[str, Any]):
        """Save configuration."""
        save_json(config, self.run_dir / 'config.json')
        
    def log_final_metrics(self, metrics: Dict[str, Any]):
        """Log final metrics and save to CSV."""
        self.log_metrics(metrics)
        df = pd.DataFrame(self.metrics)
        save_csv(df, self.run_dir / 'metrics.csv')
        save_json(metrics, self.run_dir / 'final_metrics.json')
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of logged metrics."""
        if not self.metrics:
            return {}
        return self.metrics[-1]  # Return most recent entry
