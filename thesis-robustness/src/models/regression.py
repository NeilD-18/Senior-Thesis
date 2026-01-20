"""Regression models."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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
    """Create linear regression."""
    return LinearRegression(**kwargs)


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
