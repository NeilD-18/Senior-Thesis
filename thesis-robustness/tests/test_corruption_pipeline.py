"""Test corruption pipeline integration without requiring datasets."""
import numpy as np
from scipy import sparse
import yaml
from pathlib import Path

from src.corruptions import add_noise, add_missingness, create_class_imbalance, token_dropout
from src.pipelines.corruption import apply_corruption


def test_apply_corruption():
    """Test the apply_corruption function."""
    print("Testing apply_corruption function...")
    
    # Test additive noise
    X = np.random.randn(100, 10)
    config = {'type': 'additive_noise', 'severity': 0.3, 'noise_type': 'gaussian'}
    X_corrupted, y_corrupted = apply_corruption(X, None, config, random_state=42)
    assert X_corrupted.shape == X.shape, "Shape should be preserved"
    assert y_corrupted is None, "y should be None for noise"
    print("  ✓ Additive noise via apply_corruption")
    
    # Test missingness
    config = {'type': 'missingness', 'severity': 0.2}
    X_corrupted, y_corrupted = apply_corruption(X, None, config, random_state=42)
    assert np.isnan(X_corrupted).sum() > 0, "Should have missing values"
    print("  ✓ Missingness via apply_corruption")
    
    # Test class imbalance
    y = np.concatenate([np.zeros(500), np.ones(500)])
    X = np.random.randn(1000, 10)
    config = {'type': 'class_imbalance', 'severity': 0.1, 'minority_class': 1}
    X_corrupted, y_corrupted = apply_corruption(X, y, config, random_state=42)
    assert y_corrupted is not None, "y should be returned"
    assert len(y_corrupted) < len(y), "Should have fewer samples"
    print("  ✓ Class imbalance via apply_corruption")
    
    # Test token dropout
    X_sparse = sparse.random(100, 1000, density=0.1, format='csr', random_state=42)
    config = {'type': 'token_dropout', 'severity': 0.3}
    X_corrupted, y_corrupted = apply_corruption(X_sparse, None, config, random_state=42)
    assert X_corrupted.nnz < X_sparse.nnz, "Should have fewer non-zero entries"
    print("  ✓ Token dropout via apply_corruption")
    
    print("✓ apply_corruption function works correctly\n")


def test_config_parsing():
    """Test that config files can be parsed correctly."""
    print("Testing config file parsing...")
    
    # Test noise config
    config_path = Path('configs/adult_noise.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'corruption' in config, "Config should have corruption section"
        assert config['corruption']['type'] == 'additive_noise', "Should parse corruption type"
        assert config['corruption']['severity'] == 0.3, "Should parse severity"
        print("  ✓ adult_noise.yaml parses correctly")
    
    # Test missingness config
    config_path = Path('configs/adult_missingness.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config['corruption']['type'] == 'missingness', "Should parse corruption type"
        print("  ✓ adult_missingness.yaml parses correctly")
    
    # Test imbalance config
    config_path = Path('configs/adult_imbalance.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config['corruption']['type'] == 'class_imbalance', "Should parse corruption type"
        print("  ✓ adult_imbalance.yaml parses correctly")
    
    print("✓ Config files parse correctly\n")


def test_severity_scaling():
    """Test that severity scales correctly."""
    print("Testing severity scaling...")
    
    X = np.random.randn(1000, 10)
    severities = [0.0, 0.1, 0.3, 0.5, 1.0]
    
    # Test noise scaling
    stds = []
    for s in severities:
        X_c = add_noise(X, severity=s, random_state=42)
        stds.append(np.mean(np.std(X_c, axis=0)))
    
    # Should be monotonic
    assert stds == sorted(stds), "Std should increase with severity"
    print("  ✓ Noise severity scales correctly")
    
    # Test missingness scaling
    missing_counts = []
    for s in severities:
        X_c = add_missingness(X, severity=s, random_state=42)
        missing_counts.append(np.isnan(X_c).sum())
    
    assert missing_counts == sorted(missing_counts), "Missing count should increase with severity"
    print("  ✓ Missingness severity scales correctly")
    
    print("✓ Severity scaling works correctly\n")


if __name__ == '__main__':
    print("="*60)
    print("CORRUPTION PIPELINE INTEGRATION TESTS")
    print("="*60)
    print()
    
    try:
        test_apply_corruption()
        test_config_parsing()
        test_severity_scaling()
        
        print("="*60)
        print("✓ ALL PIPELINE TESTS PASSED")
        print("="*60)
        print("\nNote: End-to-end dataset tests require network access")
        print("      to download datasets. Corruption modules are working correctly!")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
