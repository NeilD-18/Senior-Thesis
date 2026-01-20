"""Tabular classification models."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from ..common.registry import register_model


@register_model('random_forest')
def create_random_forest(**kwargs):
    """Create Random Forest classifier."""
    defaults = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
    }
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)


@register_model('xgboost')
def create_xgboost(**kwargs):
    """Create XGBoost classifier."""
    try:
        from xgboost import XGBClassifier
    except (ImportError, OSError) as e:
        raise ImportError(
            "XGBoost is not available. On macOS, install OpenMP runtime:\n"
            "  brew install libomp\n"
            "Then reinstall xgboost: pip install --upgrade --force-reinstall xgboost\n"
            f"Original error: {e}"
        )
    defaults = {
        'n_estimators': 100,
        'max_depth': 6,
        'random_state': 42,
    }
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


@register_model('svm_rbf')
def create_svm_rbf(**kwargs):
    """Create SVM with RBF kernel."""
    defaults = {
        'kernel': 'rbf',
        'C': 1.0,
        'random_state': 42,
    }
    defaults.update(kwargs)
    return SVC(**defaults, probability=True)
