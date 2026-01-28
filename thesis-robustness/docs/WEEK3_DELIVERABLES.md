# Week 3 Deliverables: Experimental Design Finalization

## Completed Tasks

### 1. Corruption Modules ✅

All corruption modules have been implemented and validated:

#### Tabular Corruptions (`src/corruptions/tabular.py`)
- **Additive Noise**: Zero-mean Gaussian/uniform noise injection
  - Parameterized by severity (fraction of feature std)
  - Supports feature masking for selective corruption
  - Handles both dense and sparse matrices
  
- **Missingness**: Random entry masking
  - Parameterized by severity (fraction of entries to mask)
  - Configurable missing value representation
  - Supports feature masking
  
- **Class Imbalance**: Controlled subsampling
  - Parameterized by severity (target imbalance ratio)
  - Configurable minority class label
  - Maintains original data order

#### Text Corruptions (`src/corruptions/text.py`)
- **Token Dropout**: Random token removal for TF-IDF matrices
  - Parameterized by severity (fraction of tokens to drop)
  - Preserves sparse matrix format
  - Maintains matrix dimensions

### 2. Corruption Pipeline ✅

Created `src/pipelines/corruption.py` with:
- `apply_corruption()`: Generic corruption application function
- `run_corruption_experiment()`: Full experiment pipeline with corruption
- Integration with existing baseline pipeline
- Automatic missing value imputation when needed
- Comprehensive logging and metrics

### 3. Configuration Schemas ✅

Extended configuration format to support corruption:
```yaml
corruption:
  type: <corruption_type>
  severity: <float in [0, 1]>
  # Type-specific parameters
```

Created example config files:
- `configs/adult_noise.yaml`
- `configs/adult_missingness.yaml`
- `configs/adult_imbalance.yaml`
- `configs/imdb_token_dropout.yaml`
- `configs/airbnb_noise.yaml`
- `configs/airbnb_missingness.yaml`

### 4. CLI Tools ✅

- `src/cli/run_corruption.py`: Run single corruption experiment
- `src/cli/run_severity_grid.py`: Run experiments across severity grid
  - Supports custom severity lists or evenly-spaced grids
  - Generates summary YAML with all results

### 5. Random Seed Control ✅

- All corruption functions accept `random_state` parameter
- Seed management via `common.seed.set_seed()` ensures reproducibility
- Seeds propagated through entire pipeline

### 6. Evaluation Procedures ✅

- Corruption applied to training data only (by default)
- Test/validation sets remain clean for fair evaluation
- Metrics computed identically to baseline experiments
- Results logged with corruption metadata

### 7. Registry System ✅

- Corruption functions registered via decorator pattern
- Extensible system for adding new corruption types
- Consistent API across all corruption types

## Testing

Created `test_corruptions.py` to validate:
- Additive noise increases variance correctly
- Missingness creates expected fraction of NaN values
- Class imbalance produces target ratio
- Token dropout reduces non-zero entries appropriately
- Reproducibility with same random seed

## Usage Examples

### Single Experiment
```bash
python -m src.cli.run_corruption --config configs/adult_noise.yaml
```

### Severity Grid
```bash
python -m src.cli.run_severity_grid \
    --config configs/adult_noise.yaml \
    --min-severity 0.0 \
    --max-severity 1.0 \
    --n-points 11
```

### Custom Severities
```bash
python -m src.cli.run_severity_grid \
    --config configs/adult_noise.yaml \
    --severities "0.0,0.1,0.2,0.3,0.5,0.7,1.0"
```

## File Structure

```
src/
├── corruptions/
│   ├── __init__.py          # Module exports
│   ├── registry.py           # Registration system
│   ├── tabular.py            # Tabular corruptions
│   └── text.py               # Text corruptions
├── pipelines/
│   ├── baseline.py           # Original baseline pipeline
│   └── corruption.py         # Corruption pipeline
└── cli/
    ├── run_baseline.py       # Baseline CLI
    ├── run_corruption.py      # Corruption CLI
    └── run_severity_grid.py  # Severity grid CLI

configs/
├── adult_noise.yaml
├── adult_missingness.yaml
├── adult_imbalance.yaml
├── imdb_token_dropout.yaml
├── airbnb_noise.yaml
└── airbnb_missingness.yaml

test_corruptions.py           # Validation tests
```

## Next Steps (Week 4)

1. Run pilot experiments on Adult and IMDB datasets
2. Validate factor levels and runtime expectations
3. Confirm experimental design with advisor
4. Begin full robustness experiments

## Notes

- All corruption functions are deterministic with fixed random seeds
- Missing value handling is consistent across experiments
- Configuration files support YAML null values (converted to np.nan)
- Sparse matrix support for text/TF-IDF data
- Class imbalance only affects training data (test distribution unchanged)
