"""Evaluation metrics."""
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error
)
import numpy as np


def compute_classification_metrics(y_true, y_pred, y_proba=None):
    """Compute classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
    }
    
    if y_proba is not None:
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auroc'] = np.nan
            
    return metrics


def compute_regression_metrics(y_true, y_pred):
    """Compute regression metrics."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
    }
