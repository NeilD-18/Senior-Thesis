"""Corruption pipeline for robustness evaluation."""
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import sparse

from ..common.seed import set_seed
from ..common.split import train_val_test_split
from ..common.metrics import compute_classification_metrics, compute_regression_metrics
from ..common.registry import get_dataset, get_model
from ..common.logging import RunLogger
from ..corruptions import (
    add_noise,
    add_missingness,
    create_class_imbalance,
    token_dropout,
    get_corruption,
)


def apply_corruption(
    X: np.ndarray,
    y: Optional[np.ndarray],
    corruption_config: Dict[str, Any],
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply corruption to data according to configuration.
    
    Args:
        X: Feature matrix
        y: Optional target labels (needed for class imbalance)
        corruption_config: Configuration dict with keys:
            - type: Corruption type ('additive_noise', 'missingness', 'class_imbalance', 'token_dropout')
            - severity: Severity value in [0, 1]
            - Additional parameters specific to corruption type
        random_state: Random seed
    
    Returns:
        (X_corrupted, y_corrupted): Corrupted data
    """
    corruption_type = corruption_config['type']
    severity = corruption_config.get('severity', 0.0)
    
    # Set seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Apply appropriate corruption
    if corruption_type == 'additive_noise':
        X_corrupted = add_noise(
            X,
            severity=severity,
            random_state=random_state,
            noise_type=corruption_config.get('noise_type', 'gaussian'),
            feature_mask=corruption_config.get('feature_mask', None)
        )
        y_corrupted = y
    
    elif corruption_type == 'missingness':
        missing_value = corruption_config.get('missing_value', np.nan)
        # Handle YAML null values
        if missing_value is None or missing_value == 'null':
            missing_value = np.nan
        X_corrupted = add_missingness(
            X,
            severity=severity,
            random_state=random_state,
            missing_value=missing_value,
            feature_mask=corruption_config.get('feature_mask', None)
        )
        y_corrupted = y
    
    elif corruption_type == 'class_imbalance':
        if y is None:
            raise ValueError("class_imbalance corruption requires target labels")
        X_corrupted, y_corrupted = create_class_imbalance(
            X,
            y,
            severity=severity,
            minority_class=corruption_config.get('minority_class', 1),
            random_state=random_state
        )
    
    elif corruption_type == 'token_dropout':
        if not sparse.issparse(X):
            raise ValueError("token_dropout requires sparse matrix (TF-IDF)")
        X_corrupted = token_dropout(
            X,
            severity=severity,
            random_state=random_state
        )
        y_corrupted = y
    
    else:
        # Try to get from registry
        try:
            corruption_func = get_corruption(corruption_type)
            if y is None:
                X_corrupted = corruption_func(X, severity=severity, random_state=random_state, **corruption_config)
                y_corrupted = y
            else:
                result = corruption_func(X, y, severity=severity, random_state=random_state, **corruption_config)
                if isinstance(result, tuple):
                    X_corrupted, y_corrupted = result
                else:
                    X_corrupted = result
                    y_corrupted = y
        except ValueError:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
    
    return X_corrupted, y_corrupted


def run_corruption_experiment(
    config: Dict[str, Any],
    run_name: str = None
):
    """
    Run robustness experiment with corruption.
    
    Args:
        config: Configuration dictionary with keys:
            - dataset: Dataset name
            - model: Model name
            - corruption: Corruption configuration dict
            - seed: Random seed
            - Additional keys from baseline config
        run_name: Optional run name
    
    Returns:
        Dictionary with results and metadata
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
    
    # Apply corruption to training data (if specified)
    corruption_config = config.get('corruption', {})
    if corruption_config and corruption_config.get('type') != 'none':
        corruption_type = corruption_config.get('type')
        severity = corruption_config.get('severity', 0.0)
        print(f"Applying corruption: {corruption_type} with severity {severity}")
        
        # Apply corruption to training data
        X_train_corrupted, y_train_corrupted = apply_corruption(
            X_train,
            y_train,
            corruption_config,
            random_state=seed
        )
        
        # For class imbalance, we need to use the corrupted training set
        if corruption_type == 'class_imbalance':
            X_train = X_train_corrupted
            y_train = y_train_corrupted
            print(f"After class imbalance: Train size = {X_train.shape[0]}")
        else:
            X_train = X_train_corrupted
        
        # For some corruptions, we might also corrupt validation/test
        # But by default, we only corrupt training to measure robustness
        # during training phase
    else:
        print("No corruption applied (baseline)")
    
    # Handle missing values if corruption introduced them
    has_nan = False
    if sparse.issparse(X_train):
        has_nan = np.isnan(X_train.data).any() if X_train.data.size > 0 else False
    else:
        has_nan = np.isnan(X_train).any()
    
    if has_nan:
        from sklearn.impute import SimpleImputer
        print("Imputing missing values...")
        imputer = SimpleImputer(strategy='mean')
        if sparse.issparse(X_train):
            X_train = X_train.toarray()
            X_val = X_val.toarray()
            X_test = X_test.toarray()
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)
    
    # Get model
    model_name = config['model']
    model_params = config.get('model_params', {})
    model_params['random_state'] = seed
    model = get_model(model_name, **model_params)
    
    print(f"Training model: {model_name}")
    model.fit(X_train, y_train)
    
    # Evaluate on validation and test sets
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Get probabilities if available
    y_val_proba = None
    y_test_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_val_proba = model.predict_proba(X_val)
            y_test_proba = model.predict_proba(X_test)
            if y_val_proba.shape[1] == 2:
                y_val_proba = y_val_proba[:, 1]
                y_test_proba = y_test_proba[:, 1]
        except:
            pass
    
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
        corruption_str = corruption_config.get('type', 'none')
        severity_str = f"_{corruption_config.get('severity', 0.0):.2f}" if corruption_config else ""
        run_name = f"{dataset_name}_{model_name}_{corruption_str}{severity_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger = RunLogger(output_dir / run_name)
    logger.log_config(config)
    
    # Combine metrics
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
        'run_dir': logger.run_dir,
        'corruption_config': corruption_config
    }
