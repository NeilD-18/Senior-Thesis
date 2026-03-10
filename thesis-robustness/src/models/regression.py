"""Regression models."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from ..common.registry import register_model


@register_model('random_forest_reg')
def create_random_forest_reg(**kwargs):
    """Create Random Forest regressor."""
    defaults = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
    }
    defaults.update(kwargs)
    return RandomForestRegressor(**defaults)


@register_model('linear_regression')
def create_linear_regression(**kwargs):
    """Create a stable linear-regression-style baseline.

    On some environments in this project, sklearn LinearRegression can
    stall indefinitely on Airbnb-sized fits. We use near-OLS Ridge with
    an iterative solver to preserve linear behavior while keeping runs
    tractable and reproducible.
    """
    # Keep compatibility with pipeline's injected random_state.
    random_state = kwargs.pop('random_state', 42)
    defaults = {
        'alpha': 1e-6,
        'solver': 'lsqr',
        'random_state': random_state,
    }
    defaults.update(kwargs)
    return Ridge(**defaults)


@register_model('xgboost_reg')
def create_xgboost_reg(**kwargs):
    """Create XGBoost regressor."""
    try:
        from xgboost import XGBRegressor
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
    return XGBRegressor(**defaults)
