"""Amazon Multi-Domain Sentiment Dataset loader."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from ..common.registry import register_dataset


@register_dataset('amazon')
def load_amazon(
    data_dir: Path = None, 
    domain: str = 'books',  # 'books', 'dvd', 'electronics', 'kitchen'
    max_features=5000, 
    ngram_range=(1, 2),
    **kwargs
):
    """
    Load and preprocess Amazon Multi-Domain Sentiment Dataset.
    
    Dataset source: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
    
    Expected directory structure:
        data/processed_acl/
            books/
                positive.review
                negative.review
            dvd/
                positive.review
                negative.review
            electronics/
                positive.review
                negative.review
            kitchen/
                positive.review
                negative.review
    
    The files are in processed bag-of-words format: word1:count1 word2:count2 ... #label#:positive/negative
    
    Args:
        data_dir: Base data directory (default: 'data')
        domain: Domain to load ('books', 'dvd', 'electronics', 'kitchen')
        max_features: Maximum number of TF-IDF features
        ngram_range: N-gram range for TF-IDF (will be converted to tuple if list)
    
    Returns:
        X: TF-IDF feature matrix
        y: Target labels (0=negative, 1=positive)
    """
    # Convert ngram_range from list to tuple if needed (YAML parses as list)
    if isinstance(ngram_range, list):
        ngram_range = tuple(ngram_range)
    
    if data_dir is None:
        data_dir = Path('data')
    
    # Normalize domain name
    domain = domain.lower()
    valid_domains = ['books', 'dvd', 'electronics', 'kitchen']
    if domain not in valid_domains:
        raise ValueError(f"Domain must be one of {valid_domains}, got: {domain}")
    
    # Look for processed_acl directory
    processed_dir = data_dir / 'processed_acl'
    
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Amazon dataset directory not found: {processed_dir}\n"
            "Please ensure the processed_acl folder is in the data directory."
        )
    
    domain_dir = processed_dir / domain
    
    if not domain_dir.exists():
        raise FileNotFoundError(
            f"Domain directory not found: {domain_dir}\n"
            f"Available domains: {[d.name for d in processed_dir.iterdir() if d.is_dir()]}"
        )
    
    pos_file = domain_dir / 'positive.review'
    neg_file = domain_dir / 'negative.review'
    
    if not pos_file.exists() or not neg_file.exists():
        raise FileNotFoundError(
            f"Review files not found in {domain_dir}\n"
            f"Expected: positive.review and negative.review"
        )
    
    texts = []
    labels = []
    
    def parse_review_line(line):
        """Parse a bag-of-words review line and reconstruct text."""
        line = line.strip()
        if not line:
            return None
        
        # Split into tokens and label
        parts = line.split()
        if not parts:
            return None
        
        # Extract label (last token should be #label#:positive/negative)
        label_part = parts[-1]
        if label_part.startswith('#label#'):
            label = label_part.split(':')[-1]
            tokens = parts[:-1]
        else:
            # No label found, assume positive (shouldn't happen)
            label = None
            tokens = parts
        
        # Reconstruct text from word:count pairs
        words = []
        for token in tokens:
            if ':' in token:
                word, count_str = token.rsplit(':', 1)
                try:
                    count = int(count_str)
                    # Repeat word by its count
                    words.extend([word] * count)
                except ValueError:
                    # If count is not a number, just add the word once
                    words.append(word)
            else:
                # No count, just add the word
                words.append(token)
        
        text = ' '.join(words)
        return text, label
    
    # Read positive reviews
    print(f"Loading Amazon {domain} positive reviews from {pos_file}...")
    with open(pos_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            result = parse_review_line(line)
            if result:
                text, label = result
                if label == 'positive':
                    texts.append(text)
                    labels.append(1)
    
    # Read negative reviews
    print(f"Loading Amazon {domain} negative reviews from {neg_file}...")
    with open(neg_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            result = parse_review_line(line)
            if result:
                text, label = result
                if label == 'negative':
                    texts.append(text)
                    labels.append(0)
    
    if len(texts) == 0:
        raise ValueError(f"No reviews found for domain {domain}")
    
    print(f"Loaded {len(texts)} reviews ({sum(labels)} positive, {len(labels) - sum(labels)} negative)")
    
    # Convert to numpy arrays
    texts = np.array(texts)
    y = np.array(labels)
    
    # Create TF-IDF features (using same vectorizer settings as IMDB for consistency)
    print(f"Creating TF-IDF features (max_features={max_features}, ngram_range={ngram_range})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        lowercase=True,
        min_df=2,
        max_df=0.95,
    )
    X = vectorizer.fit_transform(texts)  # Keep sparse for efficiency
    
    print(f"TF-IDF matrix shape: {X.shape}")
    
    return X, y
