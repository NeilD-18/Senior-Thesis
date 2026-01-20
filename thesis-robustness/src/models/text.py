"""Text classification models."""
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from ..common.registry import register_model


@register_model('linear_svm')
def create_linear_svm(**kwargs):
    """Create linear SVM for text."""
    defaults = {
        'C': 1.0,
        'max_iter': 5000,  # Increased for convergence
        'dual': True,  # Better for n_samples < n_features (text data)
        'random_state': 42,
    }
    defaults.update(kwargs)
    return LinearSVC(**defaults)


@register_model('logistic')
def create_logistic(**kwargs):
    """Create logistic regression for text."""
    defaults = {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42,
    }
    defaults.update(kwargs)
    return LogisticRegression(**defaults)
