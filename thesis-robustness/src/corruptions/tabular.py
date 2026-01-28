"""Tabular data corruption functions."""
import numpy as np
from typing import Tuple, Optional
from scipy import sparse
from .registry import register_corruption


@register_corruption('additive_noise')
def add_noise(
    X: np.ndarray,
    severity: float,
    random_state: Optional[int] = None,
    noise_type: str = 'gaussian',
    feature_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Add zero-mean noise to numeric features.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        severity: Noise severity in [0, 1]. 
                  For Gaussian noise, this controls the standard deviation
                  as a fraction of the feature's standard deviation.
        random_state: Random seed for reproducibility
        noise_type: Type of noise ('gaussian' or 'uniform')
        feature_mask: Boolean mask indicating which features to corrupt.
                      If None, all numeric features are corrupted.
    
    Returns:
        Corrupted feature matrix
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X_corrupted = X.copy()
    
    # Handle sparse matrices
    is_sparse = sparse.issparse(X)
    if is_sparse:
        X_corrupted = X_corrupted.toarray()
    
    # Determine which features to corrupt
    if feature_mask is None:
        feature_mask = np.ones(X_corrupted.shape[1], dtype=bool)
    
    # Compute feature-wise statistics for scaling noise
    feature_stds = np.std(X_corrupted[:, feature_mask], axis=0)
    feature_stds = np.maximum(feature_stds, 1e-8)  # Avoid division by zero
    
    # Generate noise
    n_samples, n_features = X_corrupted.shape
    n_corrupt_features = feature_mask.sum()
    
    if noise_type == 'gaussian':
        # Scale noise by feature standard deviation
        noise_scale = severity * feature_stds
        noise = np.random.normal(0, noise_scale, size=(n_samples, n_corrupt_features))
    elif noise_type == 'uniform':
        # Uniform noise scaled by feature std
        noise_scale = severity * feature_stds * np.sqrt(3)  # Match variance to Gaussian
        noise = np.random.uniform(-noise_scale, noise_scale, size=(n_samples, n_corrupt_features))
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'uniform'")
    
    # Apply noise to selected features
    X_corrupted[:, feature_mask] += noise
    
    # Convert back to sparse if original was sparse
    if is_sparse:
        X_corrupted = sparse.csr_matrix(X_corrupted)
    
    return X_corrupted


@register_corruption('missingness')
def add_missingness(
    X: np.ndarray,
    severity: float,
    random_state: Optional[int] = None,
    missing_value: float = np.nan,
    feature_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Randomly mask entries as missing according to severity.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        severity: Fraction of entries to mask (in [0, 1])
        random_state: Random seed for reproducibility
        missing_value: Value to use for missing entries (default: np.nan)
        feature_mask: Boolean mask indicating which features can have missingness.
                      If None, all features can be corrupted.
    
    Returns:
        Feature matrix with missing entries set to missing_value
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X_corrupted = X.copy()
    
    # Handle sparse matrices
    is_sparse = sparse.issparse(X)
    if is_sparse:
        X_corrupted = X_corrupted.toarray()
    
    # Determine which features can be corrupted
    if feature_mask is None:
        feature_mask = np.ones(X_corrupted.shape[1], dtype=bool)
    
    n_samples, n_features = X_corrupted.shape
    
    # Create mask for entries to corrupt
    # Severity controls the fraction of entries across all eligible features
    n_eligible_entries = n_samples * feature_mask.sum()
    n_to_corrupt = int(severity * n_eligible_entries)
    
    # Sample random entries to corrupt
    eligible_indices = []
    for i in range(n_samples):
        for j in range(n_features):
            if feature_mask[j]:
                eligible_indices.append((i, j))
    
    if len(eligible_indices) > 0:
        corrupt_indices = np.random.choice(
            len(eligible_indices),
            size=min(n_to_corrupt, len(eligible_indices)),
            replace=False
        )
        
        for idx in corrupt_indices:
            i, j = eligible_indices[idx]
            X_corrupted[i, j] = missing_value
    
    # Convert back to sparse if original was sparse
    if is_sparse:
        X_corrupted = sparse.csr_matrix(X_corrupted)
    
    return X_corrupted


@register_corruption('class_imbalance')
def create_class_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    severity: float,
    minority_class: int = 1,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create class imbalance by subsampling the minority class.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels
        severity: Target imbalance ratio. 
                  severity=0.1 means minority class will be 10% of majority class size.
                  severity=1.0 means balanced classes.
        minority_class: Label of the minority class (default: 1)
        random_state: Random seed for reproducibility
    
    Returns:
        (X_imbalanced, y_imbalanced): Subsampled data with class imbalance
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Identify minority and majority classes
    minority_mask = (y == minority_class)
    majority_mask = ~minority_mask
    
    n_minority = minority_mask.sum()
    n_majority = majority_mask.sum()
    
    if n_minority == 0 or n_majority == 0:
        # Already imbalanced or single class - return as-is
        return X.copy(), y.copy()
    
    # Calculate target minority class size
    target_minority_size = int(severity * n_majority)
    target_minority_size = max(1, min(target_minority_size, n_minority))  # Clamp to valid range
    
    # Subsample minority class
    minority_indices = np.where(minority_mask)[0]
    selected_minority_indices = np.random.choice(
        minority_indices,
        size=target_minority_size,
        replace=False
    )
    
    # Combine with all majority class samples
    majority_indices = np.where(majority_mask)[0]
    selected_indices = np.concatenate([majority_indices, selected_minority_indices])
    selected_indices = np.sort(selected_indices)  # Maintain original order
    
    # Return subsampled data
    if sparse.issparse(X):
        X_imbalanced = X[selected_indices]
    else:
        X_imbalanced = X[selected_indices]
    y_imbalanced = y[selected_indices]
    
    return X_imbalanced, y_imbalanced
