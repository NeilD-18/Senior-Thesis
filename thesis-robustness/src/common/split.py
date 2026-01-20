"""Data splitting utilities."""
from sklearn.model_selection import train_test_split
import numpy as np


def train_val_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train, validation, and test sets."""
    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y if len(np.unique(y)) < 20 else None
    )
    
    # Second split: separate validation from train
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, 
        random_state=random_state, 
        stratify=y_train_val if len(np.unique(y_train_val)) < 20 else None
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
