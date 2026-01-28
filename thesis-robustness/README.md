# ML Model Robustness Study

Empirical evaluation of machine learning model robustness under distribution shifts and data corruption.

## Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run a single corruption experiment
python -m src.cli.run_corruption --config configs/adult_noise.yaml

# 3. Run a severity grid (multiple corruption levels)
python -m src.cli.run_severity_grid \
    --config configs/adult_noise.yaml \
    --min-severity 0.0 \
    --max-severity 1.0 \
    --n-points 11

# 4. Analyze results
python -m src.cli.analyze_severity_grid \
    --summary outputs/severity_grids/severity_grid_summary.yaml
```

## Table of Contents

- [Setup](#setup)
- [Running Experiments](#running-experiments)
- [Understanding Results](#understanding-results)
- [Configuration](#configuration)
- [Available Corruptions](#available-corruptions)
- [Project Structure](#project-structure)

## Setup

### 1. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install package (if not already installed)
pip install -e .
```

### 2. Prepare Datasets

**Adult Income** (automatic):
- Automatically downloaded from UCI ML Repository on first run
- No manual setup required

**IMDB** (local files):
- Place `aclImdb 2` folder in `data/` directory
- Expected structure: `data/aclImdb 2/train/pos/` and `data/aclImdb 2/train/neg/`

**Amazon Reviews** (local files):
- Place `processed_acl` folder in `data/` directory
- Expected structure: `data/processed_acl/books/positive.review` etc.

**Airbnb** (manual via Kaggle):
```bash
# Install Kaggle CLI: pip install kaggle
# Configure API: https://www.kaggle.com/docs/api
kaggle competitions download -c airbnb-price-prediction
# Extract and place train.csv in data/raw/airbnb.csv
```

## Running Experiments

### Baseline Experiments

Run baseline models without corruption:

```bash
python -m src.cli.run_baseline --config configs/adult_baseline.yaml
python -m src.cli.run_baseline --config configs/imdb_baseline.yaml
python -m src.cli.run_baseline --config configs/airbnb_baseline.yaml
```

### Single Corruption Experiment

Run a single experiment with corruption:

```bash
python -m src.cli.run_corruption --config configs/adult_noise.yaml
```

**Output**: Results saved to `outputs/runs/<run_name>/`

### Severity Grid Experiments

Run experiments across multiple corruption severities:

```bash
# Evenly spaced grid (0.0 to 1.0, 11 points)
python -m src.cli.run_severity_grid \
    --config configs/adult_noise.yaml \
    --min-severity 0.0 \
    --max-severity 1.0 \
    --n-points 11

# Custom severity values
python -m src.cli.run_severity_grid \
    --config configs/adult_noise.yaml \
    --severities "0.0,0.1,0.2,0.3,0.5,0.7,1.0"

# Custom output directory
python -m src.cli.run_severity_grid \
    --config configs/adult_noise.yaml \
    --output-dir outputs/my_experiment
```

**Output**: 
- Individual results in `outputs/severity_grids/<run_name>/`
- Summary YAML: `outputs/severity_grids/severity_grid_summary.yaml`

### Analyzing Results

**Option 1: Simple plotting script (recommended)**
```bash
python scripts/plot_results_simple.py \
    --dir outputs/severity_grids \
    --plot outputs/severity_grids/degradation_curves.png
```

**Option 2: Full analysis script**
```bash
python -m src.cli.analyze_severity_grid \
    --summary outputs/severity_grids/severity_grid_summary.yaml \
    --plot outputs/severity_grids/degradation_curves.png
```

This generates:
- Summary table with robustness statistics
- Degradation curve plots (3 subplots: Accuracy, F1-Score, AUROC)
- Interpretation guide

**Note**: The simple plotting script reads results directly from run directories and avoids YAML serialization issues.

## Understanding Results

### Output Structure

Each experiment creates a directory with:

```
outputs/runs/<run_name>/
├── config.json          # Experiment configuration
├── final_metrics.json   # Final metrics (JSON)
└── metrics.csv          # Metrics history (CSV)
```

### Metrics Explained

**Classification Metrics:**
- **accuracy**: Overall correctness (0.85 = 85% correct)
- **f1**: Harmonic mean of precision and recall (better for imbalanced classes)
- **auroc**: Area under ROC curve (0.88 = good discrimination, 0.5 = random)

**Regression Metrics:**
- **rmse**: Root mean squared error (lower is better)
- **mae**: Mean absolute error (lower is better)

### Interpreting Severity Grid Results

A severity grid shows how performance degrades as corruption increases:

1. **Clean Performance** (severity=0.0): Baseline without corruption
2. **Degradation Curve**: Performance vs severity
3. **Robustness Indicators**:
   - **Gradual decline**: Model is robust
   - **Sharp drop**: Model is sensitive
   - **Plateau then drop**: Tolerates low corruption, fails at high levels

See [docs/RESULTS_GUIDE.md](docs/RESULTS_GUIDE.md) for detailed interpretation.

## Configuration

### Configuration File Format

```yaml
# Dataset and model
dataset: adult              # Dataset name
model: random_forest        # Model name
seed: 42                    # Random seed for reproducibility

# Data splitting
test_size: 0.2              # Test set fraction
val_size: 0.1               # Validation set fraction

# Model parameters
model_params:
  n_estimators: 100
  max_depth: 10
  random_state: 42

# Corruption (optional)
corruption:
  type: additive_noise      # Corruption type
  severity: 0.3             # Severity [0, 1]
  noise_type: gaussian      # Type-specific parameter

# Output
output_dir: outputs/runs    # Output directory
```

### Available Config Files

**Baselines:**
- `configs/adult_baseline.yaml`
- `configs/imdb_baseline.yaml`
- `configs/airbnb_baseline.yaml`

**Corruption Experiments:**
- `configs/adult_noise.yaml` - Additive noise on Adult
- `configs/adult_missingness.yaml` - Missingness on Adult
- `configs/adult_imbalance.yaml` - Class imbalance on Adult
- `configs/imdb_token_dropout.yaml` - Token dropout on IMDB
- `configs/airbnb_noise.yaml` - Additive noise on Airbnb
- `configs/airbnb_missingness.yaml` - Missingness on Airbnb

## Available Corruptions

### Tabular Corruptions

**Additive Noise** (`additive_noise`):
- Adds zero-mean Gaussian/uniform noise to numeric features
- Severity controls noise scale as fraction of feature std
- Parameters:
  - `noise_type`: `'gaussian'` or `'uniform'` (default: `'gaussian'`)

**Missingness** (`missingness`):
- Randomly masks entries as missing (NaN)
- Severity controls fraction of entries to mask
- Parameters:
  - `missing_value`: Value to use for missing (default: `np.nan`)

**Class Imbalance** (`class_imbalance`):
- Creates controlled class imbalance via subsampling
- Severity = target imbalance ratio (minority/majority)
- Parameters:
  - `minority_class`: Label of minority class (default: `1`)

### Text Corruptions

**Token Dropout** (`token_dropout`):
- Randomly drops tokens (zeros out TF-IDF features)
- Severity controls fraction of tokens to drop
- Only works with sparse matrices (TF-IDF)

## Project Structure

```
thesis-robustness/
├── configs/              # Configuration files
├── docs/                 # Documentation files
├── outputs/              # Experiment results
│   ├── runs/            # Single experiment results
│   └── severity_grids/  # Severity grid results
├── scripts/             # Utility scripts
├── src/
│   ├── cli/             # Command-line interfaces
│   ├── common/          # Shared utilities
│   ├── corruptions/     # Corruption modules
│   ├── datasets/        # Dataset loaders
│   ├── models/          # Model definitions
│   └── pipelines/      # Experiment pipelines
├── tests/               # Test files
└── README.md           # This file
```

## Testing

Verify corruption modules work correctly:

```bash
# Run unit tests
python tests/test_corruptions.py

# Run pipeline integration tests
python tests/test_corruption_pipeline.py
```

## Common Workflows

### 1. Compare Models on Same Corruption

```bash
# Random Forest
python -m src.cli.run_severity_grid --config configs/adult_noise.yaml

# XGBoost (modify config to use xgboost model)
python -m src.cli.run_severity_grid --config configs/adult_noise_xgb.yaml

# Compare results
python -m src.cli.analyze_severity_grid --summary outputs/severity_grids/severity_grid_summary.yaml
```

### 2. Compare Corruptions on Same Model

```bash
# Noise
python -m src.cli.run_severity_grid --config configs/adult_noise.yaml

# Missingness
python -m src.cli.run_severity_grid --config configs/adult_missingness.yaml

# Compare degradation patterns
```

### 3. Reproducibility

All experiments use fixed random seeds. To reproduce:

```bash
# Use same seed in config
python -m src.cli.run_corruption --config configs/adult_noise.yaml
```

## Troubleshooting

**Issue**: Module not found errors
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Verify installation
pip list | grep thesis-robustness
```

**Issue**: Dataset not found
- Check dataset paths in `src/datasets/`
- Verify data files are in correct locations
- See dataset-specific setup instructions above

**Issue**: Results not degrading with severity
- Verify corruption is being applied (check console output)
- Check that severity parameter is in config
- Verify corruption type matches dataset type

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - Get running in 5 minutes
- [Results Interpretation Guide](docs/RESULTS_GUIDE.md) - How to interpret experiment results
- [Usage Summary](docs/USAGE_SUMMARY.md) - Quick command reference
- [Documentation Index](docs/DOCUMENTATION_INDEX.md) - Complete documentation guide
- [Week 3 Deliverables](docs/WEEK3_DELIVERABLES.md) - Corruption module documentation
- [Cross-Domain Note](docs/CROSS_DOMAIN_NOTE.md) - Domain shift implementation notes

## Citation

If you use this codebase, please cite appropriately.
