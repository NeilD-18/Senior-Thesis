"""Test script to validate corruption modules."""
import numpy as np
from scipy import sparse
from src.corruptions.tabular import add_noise, add_missingness, create_class_imbalance
from src.corruptions.text import token_dropout


def test_additive_noise():
    """Test additive noise corruption."""
    print("Testing additive noise...")
    X = np.random.randn(100, 10)
    X_original_std = np.std(X, axis=0)
    
    # Test with severity 0.5
    X_corrupted = add_noise(X, severity=0.5, random_state=42)
    X_corrupted_std = np.std(X_corrupted, axis=0)
    
    # Check that std increased
    assert np.all(X_corrupted_std >= X_original_std), "Noise should increase variance"
    print("  ✓ Additive noise test passed")


def test_missingness():
    """Test missingness corruption."""
    print("Testing missingness...")
    X = np.random.randn(100, 10)
    
    # Test with severity 0.2
    X_corrupted = add_missingness(X, severity=0.2, random_state=42)
    
    # Check that some values are NaN
    n_missing = np.isnan(X_corrupted).sum()
    expected_missing = int(0.2 * X.size)
    
    assert n_missing > 0, "Should have missing values"
    assert abs(n_missing - expected_missing) < X.size * 0.1, "Missing count should be approximately correct"
    print(f"  ✓ Missingness test passed ({n_missing} missing values)")


def test_class_imbalance():
    """Test class imbalance corruption."""
    print("Testing class imbalance...")
    # Create balanced dataset
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    y = np.concatenate([np.zeros(500), np.ones(500)])
    
    # Test with severity 0.1 (minority should be 10% of majority)
    X_imbalanced, y_imbalanced = create_class_imbalance(
        X, y, severity=0.1, minority_class=1, random_state=42
    )
    
    n_minority = (y_imbalanced == 1).sum()
    n_majority = (y_imbalanced == 0).sum()
    imbalance_ratio = n_minority / n_majority
    
    assert n_minority < n_majority, "Minority class should be smaller"
    assert abs(imbalance_ratio - 0.1) < 0.05, f"Imbalance ratio should be ~0.1, got {imbalance_ratio}"
    print(f"  ✓ Class imbalance test passed (ratio: {imbalance_ratio:.3f})")


def test_token_dropout():
    """Test token dropout corruption."""
    print("Testing token dropout...")
    # Create sparse TF-IDF-like matrix
    n_samples, n_features = 100, 1000
    density = 0.1
    X = sparse.random(n_samples, n_features, density=density, format='csr', random_state=42)
    
    original_nnz = X.nnz
    
    # Test with severity 0.3
    X_corrupted = token_dropout(X, severity=0.3, random_state=42)
    
    corrupted_nnz = X_corrupted.nnz
    expected_nnz = int(original_nnz * 0.7)  # 70% remaining
    
    assert corrupted_nnz < original_nnz, "Should have fewer non-zero entries"
    assert abs(corrupted_nnz - expected_nnz) < original_nnz * 0.1, "Dropout count should be approximately correct"
    print(f"  ✓ Token dropout test passed ({original_nnz} -> {corrupted_nnz} tokens)")


def test_reproducibility():
    """Test that corruption is reproducible with same seed."""
    print("Testing reproducibility...")
    X = np.random.randn(100, 10)
    
    # Run twice with same seed
    X1 = add_noise(X, severity=0.5, random_state=42)
    X2 = add_noise(X, severity=0.5, random_state=42)
    
    assert np.allclose(X1, X2), "Results should be identical with same seed"
    print("  ✓ Reproducibility test passed")


if __name__ == '__main__':
    print("Running corruption module tests...\n")
    
    try:
        test_additive_noise()
        test_missingness()
        test_class_imbalance()
        test_token_dropout()
        test_reproducibility()
        
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise
