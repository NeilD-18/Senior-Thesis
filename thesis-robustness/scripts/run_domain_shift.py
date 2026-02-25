#!/usr/bin/env python3
"""
IMDB → Amazon domain-shift experiment.

Train on IMDB only; evaluate on IMDB test (in-domain) and Amazon (out-of-domain).
Uses a single TF-IDF vectorizer fitted on IMDB train so the feature space is consistent.
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Add project root for imports
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo))
from src.common.seed import set_seed
from src.common.metrics import compute_classification_metrics
from src.common.registry import get_model
# Trigger model registration (tabular + text)
import src.models.tabular  # noqa: F401
import src.models.text     # noqa: F401


def _find_imdb_dir(data_dir: Path) -> Path:
    for name in ['aclImdb 2', 'aclImdb', 'imdb']:
        d = data_dir / name
        if d.exists() and d.is_dir():
            return d
    raise FileNotFoundError(f"IMDB dir not found under {data_dir}")


def _load_imdb_split(imdb_dir: Path, split: str):
    """Load raw texts and labels from IMDB train or test. Returns (texts, y)."""
    split_dir = imdb_dir / split
    texts, labels = [], []
    for label_val, sub in [(1, 'pos'), (0, 'neg')]:
        dir_path = split_dir / sub
        if not dir_path.exists():
            continue
        for f in sorted(dir_path.glob('*.txt')):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    texts.append(fp.read().strip())
                    labels.append(label_val)
            except Exception:
                pass
    return np.array(texts), np.array(labels)


def _parse_amazon_line(line):
    """Parse Amazon processed_acl line; return (text, label) or None."""
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if not parts:
        return None
    label_part = parts[-1]
    if label_part.startswith('#label#'):
        label = label_part.split(':')[-1]
        tokens = parts[:-1]
    else:
        label = None
        tokens = parts
    words = []
    for token in tokens:
        if ':' in token:
            word, count_str = token.rsplit(':', 1)
            try:
                count = int(count_str)
                words.extend([word] * count)
            except ValueError:
                words.append(word)
        else:
            words.append(token)
    return ' '.join(words), label


def load_amazon_raw(data_dir: Path, domain: str = 'books'):
    """Load Amazon raw texts and labels. Returns (texts, y)."""
    processed_dir = data_dir / 'processed_acl' / domain
    if not processed_dir.exists():
        raise FileNotFoundError(f"Amazon dir not found: {processed_dir}")
    texts, labels = [], []
    for label_val, fname in [(1, 'positive.review'), (0, 'negative.review')]:
        path = processed_dir / fname
        if not path.exists():
            continue
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                out = _parse_amazon_line(line)
                if not out:
                    continue
                text, label = out
                if (label_val == 1 and label == 'positive') or (label_val == 0 and label == 'negative'):
                    texts.append(text)
                    labels.append(label_val)
    if not texts:
        raise ValueError(f"No Amazon reviews for domain {domain}")
    return np.array(texts), np.array(labels)


def run_domain_shift(
    data_dir: Path,
    output_dir: Path,
    seeds=(42, 43, 44),
    models=('linear_svm', 'random_forest', 'xgboost'),
    max_features=5000,
    ngram_range=(1, 2),
    val_frac=0.1,
    amazon_domain='books',
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading IMDB train and test (raw texts)...")
    imdb_dir = _find_imdb_dir(data_dir)
    texts_train_imdb, y_train_full = _load_imdb_split(imdb_dir, 'train')
    texts_test_imdb, y_test_imdb = _load_imdb_split(imdb_dir, 'test')
    print("Loading Amazon (raw texts)...")
    texts_amazon, y_amazon = load_amazon_raw(data_dir, domain=amazon_domain)

    print("Fitting TF-IDF on IMDB train...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        lowercase=True,
        min_df=2,
        max_df=0.95,
    )
    X_train_full = vectorizer.fit_transform(texts_train_imdb)
    X_test_imdb = vectorizer.transform(texts_test_imdb)
    X_amazon = vectorizer.transform(texts_amazon)
    print(f"Shapes: X_train_full={X_train_full.shape}, X_test_imdb={X_test_imdb.shape}, X_amazon={X_amazon.shape}")

    all_results = []
    for model_name in models:
        for seed in seeds:
            set_seed(seed)
            # Train/val split from IMDB train
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=val_frac, random_state=seed, stratify=y_train_full
            )
            model = get_model(model_name, random_state=seed)
            model.fit(X_train, y_train)

            def get_proba(model, X):
                if hasattr(model, 'predict_proba'):
                    try:
                        p = model.predict_proba(X)
                        return p[:, 1] if p.shape[1] == 2 else p
                    except Exception:
                        pass
                if hasattr(model, 'decision_function'):
                    return model.decision_function(X)
                return None

            y_val_pred = model.predict(X_val)
            y_test_imdb_pred = model.predict(X_test_imdb)
            y_amazon_pred = model.predict(X_amazon)
            y_val_proba = get_proba(model, X_val)
            y_test_imdb_proba = get_proba(model, X_test_imdb)
            y_amazon_proba = get_proba(model, X_amazon)

            val_metrics = compute_classification_metrics(y_val, y_val_pred, y_val_proba)
            test_imdb_metrics = compute_classification_metrics(y_test_imdb, y_test_imdb_pred, y_test_imdb_proba)
            amazon_metrics = compute_classification_metrics(y_amazon, y_amazon_pred, y_amazon_proba)

            run = {
                'model': model_name,
                'seed': seed,
                'val_metrics': val_metrics,
                'imdb_test_metrics': test_imdb_metrics,
                'amazon_metrics': amazon_metrics,
            }
            all_results.append(run)
            run_dir = output_dir / f"{model_name}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / 'metrics.json', 'w') as f:
                json.dump({k: v for k, v in run.items() if k != 'model'}, f, indent=2)
            print(f"  {model_name} seed={seed} IMDB_test acc={test_imdb_metrics.get('accuracy', 0):.4f} Amazon acc={amazon_metrics.get('accuracy', 0):.4f}")

    # Aggregate per model (mean ± std over seeds)
    by_model = {}
    for r in all_results:
        name = r['model']
        if name not in by_model:
            by_model[name] = []
        by_model[name].append(r)

    summary = {}
    for name, runs in by_model.items():
        for key in ['imdb_test_metrics', 'amazon_metrics']:
            prefix = key.replace('_metrics', '')
            summary[f"{name}_{prefix}"] = {}
            for m in ['accuracy', 'f1', 'auroc']:
                vals = [run[key].get(m) for run in runs if run[key].get(m) is not None]
                if vals:
                    summary[f"{name}_{prefix}"][f"{m}_mean"] = float(np.mean(vals))
                    summary[f"{name}_{prefix}"][f"{m}_std"] = float(np.std(vals))

    with open(output_dir / 'domain_shift_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {output_dir / 'domain_shift_summary.json'}")
    return all_results, summary


def main():
    ap = argparse.ArgumentParser(description='IMDB → Amazon domain-shift experiment')
    ap.add_argument('--data-dir', type=str, default='data', help='Data directory')
    ap.add_argument('--output-dir', type=str, default='outputs/week7/imdb_to_amazon', help='Output directory')
    ap.add_argument('--seeds', type=str, default='42,43,44', help='Comma-separated seeds')
    ap.add_argument('--models', type=str, default='linear_svm,random_forest,xgboost', help='Comma-separated model names')
    ap.add_argument('--amazon-domain', type=str, default='books', help='Amazon domain (books, dvd, etc.)')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    models = [m.strip() for m in args.models.split(',')]
    run_domain_shift(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        seeds=tuple(seeds),
        models=tuple(models),
        amazon_domain=args.amazon_domain,
    )


if __name__ == '__main__':
    main()
