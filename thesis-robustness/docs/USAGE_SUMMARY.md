# Usage Summary

Quick reference for common commands and workflows.

## Basic Commands

### Activate Environment
```bash
source venv/bin/activate
```

### Run Single Experiment
```bash
python -m src.cli.run_corruption --config configs/adult_noise.yaml
```

### Run Severity Grid
```bash
python -m src.cli.run_severity_grid \
    --config configs/adult_noise.yaml \
    --min-severity 0.0 \
    --max-severity 1.0 \
    --n-points 11
```

### Analyze Results
```bash
python -m src.cli.analyze_severity_grid \
    --summary outputs/severity_grids/severity_grid_summary.yaml
```

## Experiment Types

### Baseline (No Corruption)
```bash
python -m src.cli.run_baseline --config configs/adult_baseline.yaml
```

### Corruption Experiments
```bash
# Additive noise
python -m src.cli.run_corruption --config configs/adult_noise.yaml

# Missingness
python -m src.cli.run_corruption --config configs/adult_missingness.yaml

# Class imbalance
python -m src.cli.run_corruption --config configs/adult_imbalance.yaml

# Token dropout (text)
python -m src.cli.run_corruption --config configs/imdb_token_dropout.yaml
```

## Output Locations

- **Single experiments**: `outputs/runs/<run_name>/`
- **Severity grids**: `outputs/severity_grids/<run_name>/`
- **Summary files**: `outputs/severity_grids/severity_grid_summary.yaml`

## Key Files

- **Configs**: `configs/*.yaml`
- **Results**: `outputs/runs/` or `outputs/severity_grids/`
- **Metrics**: `final_metrics.json` in each run directory
- **Summary**: `severity_grid_summary.yaml` for grids

## Testing

```bash
# Test corruption modules
python tests/test_corruptions.py

# Test pipeline integration
python test_corruption_pipeline.py
```

## Documentation

- **Quick Start**: `QUICKSTART.md`
- **Full Guide**: `../README.md`
- **Results Guide**: `RESULTS_GUIDE.md`
