"""Adult Income dataset loader."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from ..common.registry import register_dataset


@register_dataset('adult')
def load_adult(
    data_dir: Path = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    **kwargs
):
    """
    Load and preprocess Adult Income dataset using UCI ML Repository.
    
    Dataset source: https://archive.ics.uci.edu/dataset/2/adult
    
    Returns:
        X: Feature matrix
        y: Target labels
    """
    if data_dir is None:
        data_dir = Path('data')
    cache_file = data_dir / 'cache' / 'adult_preprocessed.npz'

    if use_cache and not refresh_cache and cache_file.exists():
        print(f"Loading Adult dataset from local cache: {cache_file}")
        with np.load(cache_file) as cached:
            return cached['X'], cached['y']

    try:
        from ucimlrepo import fetch_ucirepo
        
        # Fetch dataset from UCI ML Repository
        print("Fetching Adult dataset from UCI ML Repository...")
        adult = fetch_ucirepo(id=2)
        
        # Get features and targets as pandas DataFrames
        X = adult.data.features
        y = adult.data.targets
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.copy()
        if isinstance(y, pd.DataFrame):
            # Target is typically a single column
            y = y.iloc[:, 0].values if len(y.columns) > 0 else y.values.flatten()
        
        print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
    except ImportError:
        raise ImportError(
            "ucimlrepo package required. Install with: pip install ucimlrepo"
        )
    except Exception as e:
        if use_cache and cache_file.exists():
            print(f"Fetch failed ({e}); falling back to local cache: {cache_file}")
            with np.load(cache_file) as cached:
                return cached['X'], cached['y']
        raise RuntimeError(f"Failed to fetch Adult dataset: {e}")
    
    # Handle missing values - replace '?' with NaN
    if isinstance(X, pd.DataFrame):
        X = X.replace('?', np.nan)
        # Drop rows with missing values for baseline
        missing_mask = X.isna().any(axis=1)
        X = X[~missing_mask]
        y = y[~missing_mask] if hasattr(y, '__len__') else y
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Convert to numpy
        X = X.values.astype(float)
    
    # Ensure y is numpy array
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    # Handle binary target encoding if needed
    if y.dtype == object or isinstance(y[0], str):
        y = (y == '>50K').astype(int) if isinstance(y[0], str) else y

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, X=X, y=y)
        print(f"Cached preprocessed Adult dataset to: {cache_file}")
    
    return X, y
