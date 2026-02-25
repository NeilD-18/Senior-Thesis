"""IMDB dataset loader."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from ..common.registry import register_dataset


@register_dataset('imdb')
def load_imdb(
    data_dir: Path = None,
    max_features=5000,
    ngram_range=(1, 2),
    use_train=True,
    vectorize=True,
    **kwargs
):
    """
    Load and preprocess IMDB dataset from local directory.
    
    Expected directory structure:
        data/aclImdb 2/
            train/
                pos/  (12500 .txt files)
                neg/  (12500 .txt files)
            test/
                pos/  (12500 .txt files)
                neg/  (12500 .txt files)
    
    Args:
        data_dir: Base data directory (default: 'data')
        max_features: Maximum number of TF-IDF features
        ngram_range: N-gram range for TF-IDF (will be converted to tuple if list)
        use_train: If True, use train set; if False, use test set
        vectorize: If True, return TF-IDF features. If False, return raw text.
    
    Returns:
        X: TF-IDF feature matrix
        y: Target labels (0=negative, 1=positive)
    """
    # Convert ngram_range from list to tuple if needed (YAML parses as list)
    if isinstance(ngram_range, list):
        ngram_range = tuple(ngram_range)
    
    if data_dir is None:
        data_dir = Path('data')
    
    # Look for IMDB directory (handle different possible names)
    possible_dirs = [
        data_dir / 'aclImdb 2',
        data_dir / 'aclImdb',
        data_dir / 'imdb',
    ]
    
    imdb_dir = None
    for dir_path in possible_dirs:
        if dir_path.exists() and dir_path.is_dir():
            imdb_dir = dir_path
            break
    
    if imdb_dir is None:
        raise FileNotFoundError(
            f"IMDB dataset directory not found. Looked in: {[str(d) for d in possible_dirs]}\n"
            "Please ensure the aclImdb folder is in the data directory."
        )
    
    # Determine which split to use
    split = 'train' if use_train else 'test'
    split_dir = imdb_dir / split
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    # Read positive reviews
    pos_dir = split_dir / 'pos'
    neg_dir = split_dir / 'neg'
    
    if not pos_dir.exists() or not neg_dir.exists():
        raise FileNotFoundError(
            f"pos/ or neg/ directories not found in {split_dir}"
        )
    
    texts = []
    labels = []
    
    # Read positive reviews (label = 1)
    print(f"Loading positive reviews from {pos_dir}...")
    pos_files = sorted(pos_dir.glob('*.txt'))
    for file_path in pos_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                texts.append(text)
                labels.append(1)
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    # Read negative reviews (label = 0)
    print(f"Loading negative reviews from {neg_dir}...")
    neg_files = sorted(neg_dir.glob('*.txt'))
    for file_path in neg_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                texts.append(text)
                labels.append(0)
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    print(f"Loaded {len(texts)} reviews ({sum(labels)} positive, {len(labels) - sum(labels)} negative)")
    
    # Convert to numpy arrays
    texts = np.array(texts)
    y = np.array(labels)
    
    if not vectorize:
        print(f"Returning raw text split: {texts.shape}")
        return texts, y

    # Create TF-IDF features
    print(f"Creating TF-IDF features (max_features={max_features}, ngram_range={ngram_range})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        lowercase=True,
        min_df=2,  # Ignore terms that appear in fewer than 2 documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of documents
    )
    X = vectorizer.fit_transform(texts)  # Keep sparse for efficiency

    print(f"TF-IDF matrix shape: {X.shape}")

    return X, y
