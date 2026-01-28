# Quick Start Guide

Get up and running with the robustness study in 5 minutes.

## Step 1: Activate Environment

```bash
cd thesis-robustness
source venv/bin/activate
```

## Step 2: Verify Setup

```bash
# Test corruption modules
python tests/test_corruptions.py

# Should see: ✓ All tests passed!
```

## Step 3: Run Your First Experiment

### Option A: Single Corruption Experiment

```bash
python -m src.cli.run_corruption --config configs/adult_noise.yaml
```

**What happens:**
- Loads Adult dataset (auto-downloads if needed)
- Applies additive noise corruption (severity=0.3)
- Trains Random Forest model
- Evaluates on test set
- Saves results to `outputs/runs/`

**Output:**
```
Results:
Validation: {'accuracy': 0.86, 'f1': 0.84, ...}
Test: {'accuracy': 0.85, 'f1': 0.83, ...}

Results saved to: outputs/runs/adult_random_forest_additive_noise_0.30_<timestamp>/
```

### Option B: Severity Grid (Recommended)

```bash
python -m src.cli.run_severity_grid \
    --config configs/adult_noise.yaml \
    --min-severity 0.0 \
    --max-severity 1.0 \
    --n-points 11
```

**What happens:**
- Runs 11 experiments (severity 0.0 to 1.0)
- Shows progress bar
- Saves individual results + summary

**Output:**
```
Running experiments for 11 severity values: [0.0, 0.1, ..., 1.0]
Severity grid: 100%|████████| 11/11 [01:25<00:00]

Severity grid summary saved to: outputs/severity_grids/severity_grid_summary.yaml
Completed 11/11 experiments
```

## Step 4: Analyze Results

```bash
python -m src.cli.analyze_severity_grid \
    --summary outputs/severity_grids/severity_grid_summary.yaml \
    --plot outputs/severity_grids/degradation_curves.png
```

**What you get:**
- Summary table with all metrics
- Robustness statistics (AUC, drop, slope)
- Degradation curve plots
- Interpretation guide

## Common Tasks

### Compare Different Models

```bash
# Random Forest
python -m src.cli.run_severity_grid --config configs/adult_noise.yaml

# XGBoost (create config first or modify existing)
# Edit configs/adult_noise.yaml: change model to 'xgboost'
python -m src.cli.run_severity_grid --config configs/adult_noise.yaml
```

### Compare Different Corruptions

```bash
# Noise
python -m src.cli.run_severity_grid --config configs/adult_noise.yaml

# Missingness
python -m src.cli.run_severity_grid --config configs/adult_missingness.yaml

# Class Imbalance
python -m src.cli.run_severity_grid --config configs/adult_imbalance.yaml
```

### Run Baseline (No Corruption)

```bash
python -m src.cli.run_baseline --config configs/adult_baseline.yaml
```

## Understanding Output

### Single Experiment

Check `outputs/runs/<run_name>/final_metrics.json`:
```json
{
  "val_accuracy": 0.866,
  "test_accuracy": 0.855,
  "test_f1": 0.829
}
```

### Severity Grid

Check `outputs/severity_grids/severity_grid_summary.yaml`:
- Contains metrics for each severity level
- Use with analysis script to generate plots

## Next Steps

1. **Read the full README**: `../README.md` for complete documentation
2. **Understand results**: `RESULTS_GUIDE.md` for interpretation
3. **Explore configs**: Modify `../configs/*.yaml` to customize experiments
4. **Run more experiments**: Try different datasets, models, corruptions

## Troubleshooting

**Problem**: Module not found
```bash
# Make sure venv is activated
source venv/bin/activate
```

**Problem**: Dataset not found
- Adult: Auto-downloads (needs internet)
- IMDB/Amazon: Check `data/` directory
- Airbnb: Download from Kaggle

**Problem**: Results not degrading
- Check console output for "Applying corruption: ..."
- Verify severity parameter in config
- Check that corruption type matches dataset

## Getting Help

- See `../README.md` for detailed documentation
- See `RESULTS_GUIDE.md` for result interpretation
- Check config files in `../configs/` for examples
