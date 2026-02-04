# Week 4: Pilot Experiments and Validation

## Overview

**Week 4 Focus:** Run pilot experiments to validate factor levels, runtime expectations, and experimental design before full-scale robustness experiments.

**Deliverables:**
- Pilot results summary
- Confirmed experimental design
- Validated factor levels and runtime expectations

## Week 4 Tasks

### 1. Pilot Grids on Adult Dataset ✅

Run small-scale severity grids to validate:
- Corruption types work correctly
- Severity levels produce expected degradation
- Runtime is acceptable

**Experiments to Run:**

```bash
# 1. Additive Noise Pilot (11 severities: 0.0 to 1.0)
python -m src.cli.run_severity_grid \
    --config configs/adult_noise.yaml \
    --min-severity 0.0 \
    --max-severity 1.0 \
    --n-points 11

# 2. Missingness Pilot
python -m src.cli.run_severity_grid \
    --config configs/adult_missingness.yaml \
    --min-severity 0.0 \
    --max-severity 1.0 \
    --n-points 11

# 3. Class Imbalance Pilot
python -m src.cli.run_severity_grid \
    --config configs/adult_imbalance.yaml \
    --min-severity 0.0 \
    --max-severity 1.0 \
    --n-points 11
```

**Expected Runtime:** ~30-60 minutes per grid (11 experiments × 2-5 min each)

**Validation Checklist:**
- [ ] Performance degrades smoothly with severity
- [ ] No unexpected errors or crashes
- [ ] Results are reproducible (same seed = same results)
- [ ] Degradation curves look reasonable
- [ ] Runtime is acceptable for full-scale experiments

### 2. Pilot Grids on IMDB Dataset ✅

Run token dropout experiments:

```bash
# Token Dropout Pilot
python -m src.cli.run_severity_grid \
    --config configs/imdb_token_dropout.yaml \
    --min-severity 0.0 \
    --max-severity 1.0 \
    --n-points 11
```

**Validation Checklist:**
- [ ] Token dropout works correctly on TF-IDF matrices
- [ ] Performance degrades as expected
- [ ] Sparse matrix format is preserved
- [ ] Runtime is acceptable

### 3. Initial Airbnb Experiments ✅

Run noise and missingness pilots:

```bash
# Noise Pilot
python -m src.cli.run_severity_grid \
    --config configs/airbnb_noise.yaml \
    --min-severity 0.0 \
    --max-severity 1.0 \
    --n-points 11

# Missingness Pilot
python -m src.cli.run_severity_grid \
    --config configs/airbnb_missingness.yaml \
    --min-severity 0.0 \
    --max-severity 1.0 \
    --n-points 11
```

**Validation Checklist:**
- [ ] Regression metrics (RMSE, MAE) behave reasonably
- [ ] Log-transformed target handling is correct
- [ ] Missing value imputation works correctly

## Analyzing Pilot Results

After running pilots, analyze results:

```bash
# Generate degradation curves
python scripts/plot_results_simple.py \
    --dir outputs/severity_grids \
    --plot outputs/severity_grids/pilot_degradation_curves.png

# Or use the analysis script
python -m src.cli.analyze_severity_grid \
    --summary outputs/severity_grids/severity_grid_summary.yaml
```

## What to Validate

### Factor Levels
- **Severity range:** Is 0.0-1.0 appropriate? Should we use different ranges?
- **Number of points:** Is 11 points enough resolution? Too many?
- **Severity spacing:** Should we use log scale or custom points?

### Runtime Expectations
- **Single experiment:** How long does one experiment take?
- **Full grid:** How long for 11 severities?
- **Full factorial:** Estimate time for all models × corruptions × severities

### Experimental Design
- **Corruption application:** Is corruption applied correctly (training only)?
- **Metrics:** Are all metrics computed correctly?
- **Reproducibility:** Do results match when re-run with same seed?

### Model Selection
- **Random Forest:** Working correctly?
- **XGBoost:** Available and working?
- **SVM:** Linear and RBF kernels available?

## Pilot Results Summary Template

Create `docs/WEEK4_PILOT_SUMMARY.md` with:

```markdown
# Week 4 Pilot Results Summary

## Adult Dataset Pilots

### Additive Noise
- Clean performance (severity=0): [accuracy, f1, auroc]
- Worst performance (severity=1): [accuracy, f1, auroc]
- Degradation pattern: [gradual/sharp/plateau]
- Runtime: [X minutes for 11 severities]

### Missingness
- [Same format]

### Class Imbalance
- [Same format]

## IMDB Dataset Pilots

### Token Dropout
- [Results]

## Airbnb Dataset Pilots

### Noise
- Clean RMSE: [value]
- Worst RMSE: [value]
- Degradation pattern: [description]

### Missingness
- [Results]

## Validation Decisions

- Severity range: [confirmed/changed]
- Number of severity points: [confirmed/changed]
- Model selection: [confirmed/changed]
- Runtime acceptable: [yes/no, estimate for full experiments]
```

## Next Steps After Week 4

Once pilots are validated:
1. **Confirm experimental design** with advisor
2. **Begin full robustness experiments** (Week 5-7)
3. **Run stability analysis** (Week 5)
4. **Execute full factorial experiments** (Week 6-7)

## Troubleshooting

**Issue:** Experiments taking too long
- **Solution:** Reduce `n_points` for pilots (e.g., use 5 points: 0.0, 0.25, 0.5, 0.75, 1.0)

**Issue:** Unexpected degradation patterns
- **Solution:** Check corruption implementation, verify severity scaling

**Issue:** Memory errors
- **Solution:** Reduce dataset size for pilots, check sparse matrix handling

**Issue:** Results not reproducible
- **Solution:** Verify seed propagation, check random_state parameters

## Quick Reference

**Run single pilot:**
```bash
python -m src.cli.run_severity_grid --config configs/adult_noise.yaml --n-points 5
```

**Analyze results:**
```bash
python scripts/plot_results_simple.py --dir outputs/severity_grids
```

**Check runtime:**
```bash
time python -m src.cli.run_severity_grid --config configs/adult_noise.yaml --n-points 5
```
