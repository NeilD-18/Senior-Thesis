"""Text data corruption functions."""
import numpy as np
from scipy import sparse
from typing import Optional
from .registry import register_corruption


@register_corruption('token_dropout')
def token_dropout(
    X: sparse.spmatrix,
    severity: float,
    random_state: Optional[int] = None
) -> sparse.spmatrix:
    """
    Randomly drop tokens (set TF-IDF features to zero) according to severity.
    
    This corruption simulates incomplete text, typos, or loss of informative terms.
    For TF-IDF matrices, this corresponds to randomly zeroing out features.
    
    Args:
        X: Sparse TF-IDF feature matrix (n_samples, n_features)
        severity: Fraction of non-zero entries to drop (in [0, 1])
        random_state: Random seed for reproducibility
    
    Returns:
        Corrupted sparse feature matrix with tokens dropped
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if not sparse.issparse(X):
        raise ValueError("token_dropout expects a sparse matrix (TF-IDF representation)")
    
    # Convert to COO format for efficient manipulation
    X_coo = X.tocoo()
    
    # Calculate number of tokens to drop
    n_tokens = X_coo.nnz  # Number of non-zero entries
    n_to_drop = int(severity * n_tokens)
    n_to_drop = min(n_to_drop, n_tokens)  # Can't drop more than exist
    
    if n_to_drop == 0:
        return X.copy()
    
    # Randomly select tokens to drop
    token_indices = np.arange(n_tokens)
    drop_indices = np.random.choice(token_indices, size=n_to_drop, replace=False)
    
    # Create mask for tokens to keep
    keep_mask = np.ones(n_tokens, dtype=bool)
    keep_mask[drop_indices] = False
    
    # Create new sparse matrix with dropped tokens
    new_row = X_coo.row[keep_mask]
    new_col = X_coo.col[keep_mask]
    new_data = X_coo.data[keep_mask]
    
    # Reconstruct sparse matrix
    X_corrupted = sparse.coo_matrix(
        (new_data, (new_row, new_col)),
        shape=X.shape
    )
    
    # Convert back to original format
    if sparse.isspmatrix_csr(X):
        X_corrupted = X_corrupted.tocsr()
    elif sparse.isspmatrix_csc(X):
        X_corrupted = X_corrupted.tocsc()
    
    return X_corrupted
