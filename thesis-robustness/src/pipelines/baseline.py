"""Baseline training and evaluation pipeline."""
from pathlib import Path
import numpy as np
from typing import Dict, Any

# Import datasets and models to trigger registration
from .. import datasets, models

from ..common.seed import set_seed
from ..common.split import train_val_test_split
from ..common.metrics import compute_classification_metrics, compute_regression_metrics
from ..common.registry import get_dataset, get_model
from ..common.logging import RunLogger


def run_baseline(config: Dict[str, Any], run_name: str = None):
    """
    Run baseline experiment.
    
    Args:
        config: Configuration dictionary
        run_name: Optional run name (defaults to timestamp)
    """
    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Get dataset
    dataset_name = config['dataset']
    print(f"Loading dataset: {dataset_name}")
    X, y = get_dataset(dataset_name, **config.get('preprocessing', {}))
    print(f"Dataset shape: {X.shape}, Target shape: {y.shape}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y,
        test_size=config.get('test_size', 0.2),
        val_size=config.get('val_size', 0.1),
        random_state=seed
    )
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Get model
    model_name = config['model']
    model_params = config.get('model_params', {})
    model_params['random_state'] = seed  # Ensure reproducibility
    model = get_model(model_name, **model_params)
    
    print(f"Training model: {model_name}")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    y_val_proba = None
    
    # Try to get probability scores for AUROC
    if hasattr(model, 'predict_proba'):
        try:
            y_val_proba = model.predict_proba(X_val)
            if y_val_proba.shape[1] == 2:
                y_val_proba = y_val_proba[:, 1]
            else:
                y_val_proba = None
        except:
            y_val_proba = None
    elif hasattr(model, 'decision_function'):
        # For LinearSVC and similar models without predict_proba
        try:
            y_val_proba = model.decision_function(X_val)
        except:
            y_val_proba = None
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    y_test_proba = None
    
    # Try to get probability scores for AUROC
    if hasattr(model, 'predict_proba'):
        try:
            y_test_proba = model.predict_proba(X_test)
            if y_test_proba.shape[1] == 2:
                y_test_proba = y_test_proba[:, 1]
            else:
                y_test_proba = None
        except:
            y_test_proba = None
    elif hasattr(model, 'decision_function'):
        # For LinearSVC and similar models without predict_proba
        try:
            y_test_proba = model.decision_function(X_test)
        except:
            y_test_proba = None
    
    # Compute metrics
    is_regression = dataset_name == 'airbnb' or 'reg' in model_name
    
    if is_regression:
        val_metrics = compute_regression_metrics(y_val, y_val_pred)
        test_metrics = compute_regression_metrics(y_test, y_test_pred)
    else:
        val_metrics = compute_classification_metrics(y_val, y_val_pred, y_val_proba)
        test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_proba)
    
    # Log results
    output_dir = Path(config.get('output_dir', 'outputs/runs'))
    if run_name is None:
        from datetime import datetime
        run_name = f"{dataset_name}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger = RunLogger(output_dir / run_name)
    logger.log_config(config)
    
    # Combine val and test metrics
    all_metrics = {}
    for k, v in val_metrics.items():
        all_metrics[f'val_{k}'] = v
    for k, v in test_metrics.items():
        all_metrics[f'test_{k}'] = v
    
    logger.log_final_metrics(all_metrics)
    
    print("\nResults:")
    print("Validation:", val_metrics)
    print("Test:", test_metrics)
    
    return {
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'model': model,
        'run_dir': logger.run_dir
    }
