# Thesis Robustness Study

Empirical evaluation of ML model robustness under distribution shifts and data corruption.

## Setup

1. Install dependencies:
```bash
# Option 1: Install as package (recommended)
pip install -e .

# Option 2: Install dependencies directly
pip install numpy pandas scikit-learn xgboost pyyaml ucimlrepo matplotlib seaborn nltk tqdm
```

2. Download datasets:

   **Adult Income** (automatic via ucimlrepo):
   - No manual download needed - will fetch automatically on first run
   
   **IMDB** (local files):
   - Place the `aclImdb 2` folder in the `data/` directory
   - Expected structure: `data/aclImdb 2/train/pos/` and `data/aclImdb 2/train/neg/`
   
   **Amazon Reviews** (local files):
   - Place the `processed_acl` folder in the `data/` directory
   - Expected structure: `data/processed_acl/books/positive.review` etc.
   - Contains processed bag-of-words format reviews
   
   **Airbnb** (manual via Kaggle):
   ```bash
   # Install Kaggle CLI if needed: pip install kaggle
   # Configure Kaggle API: https://www.kaggle.com/docs/api
   kaggle competitions download -c airbnb-price-prediction
   # Extract and place train.csv in data/raw/airbnb.csv
   ```

3. Run baseline experiments:
```bash
python -m src.cli.run_baseline --config configs/adult_baseline.yaml
python -m src.cli.run_baseline --config configs/imdb_baseline.yaml
python -m src.cli.run_baseline --config configs/amazon_baseline.yaml
python -m src.cli.run_baseline --config configs/airbnb_baseline.yaml
```

4. Summarize results:
```bash
python -m src.cli.summarize --output outputs/summary/baseline_results.csv
```

## Week 1 Goals

- Run baseline models on all three datasets
- Verify preprocessing pipelines
- Ensure reproducibility
- Generate baseline accuracy/F1/RMSE tables

## Dataset Sources

- **Adult Income**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/2/adult) - Auto-downloaded via `ucimlrepo`
- **IMDB**: Local files from ACL IMDB dataset (`data/aclImdb 2/`)
- **Amazon Reviews**: Local files from Multi-Domain Sentiment Dataset (`data/processed_acl/`)
- **Airbnb**: Kaggle Competition - Manual download required
