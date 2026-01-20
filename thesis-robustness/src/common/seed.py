"""Global seed management for reproducibility."""
import random
import numpy as np
import os


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Note: sklearn models should use random_state parameter
    # XGBoost should use seed parameter
