# Project Status: Week 4 Ready

## Current Status: ✅ Week 4 Pilot Experiments

**Last Updated:** February 4, 2026

## Week 4 Objectives

According to `Winter-Term-Revised-Plan.md`:

> **Week 4: Pilot Experiments and Validation**
> - Run pilot grids on Adult and IMDB datasets
> - Conduct initial Airbnb noise and missingness experiments
> - Validate factor levels and runtime expectations
> - **Deliverables:** Pilot results summary; confirmed experimental design

## What's Ready

### ✅ Codebase Structure
- Clean, organized directory structure
- All modules implemented and tested
- CLI tools ready for pilot experiments

### ✅ Configuration Files
- **Adult:** `adult_noise.yaml`, `adult_missingness.yaml`, `adult_imbalance.yaml`
- **IMDB:** `imdb_token_dropout.yaml`
- **Airbnb:** `airbnb_noise.yaml`, `airbnb_missingness.yaml` (fixed to use `random_forest_reg`)

### ✅ Documentation
- Week 4 pilot guide: `docs/WEEK4_PILOT_GUIDE.md`
- Week 4 checklist: `docs/WEEK4_CHECKLIST.md`
- All documentation organized in `docs/` directory

### ✅ Tools & Scripts
- Severity grid runner: `src/cli/run_severity_grid.py`
- Result analyzer: `src/cli/analyze_severity_grid.py`
- Plotting script: `scripts/plot_results_simple.py`

## Next Steps

1. **Run Adult Pilots** (3 corruption types × 11 severities)
   ```bash
   python -m src.cli.run_severity_grid --config configs/adult_noise.yaml --n-points 11
   python -m src.cli.run_severity_grid --config configs/adult_missingness.yaml --n-points 11
   python -m src.cli.run_severity_grid --config configs/adult_imbalance.yaml --n-points 11
   ```

2. **Run IMDB Pilot** (token dropout × 11 severities)
   ```bash
   python -m src.cli.run_severity_grid --config configs/imdb_token_dropout.yaml --n-points 11
   ```

3. **Run Airbnb Pilots** (2 corruption types × 11 severities)
   ```bash
   python -m src.cli.run_severity_grid --config configs/airbnb_noise.yaml --n-points 11
   python -m src.cli.run_severity_grid --config configs/airbnb_missingness.yaml --n-points 11
   ```

4. **Analyze Results**
   ```bash
   python scripts/plot_results_simple.py --dir outputs/severity_grids
   ```

5. **Create Pilot Summary**
   - Document results in `docs/WEEK4_PILOT_SUMMARY.md`
   - Validate factor levels
   - Estimate full experiment runtime
   - Confirm experimental design with advisor

## Project Timeline

| Week | Status | Focus |
|------|--------|-------|
| 1 | ✅ Complete | Baseline models, preprocessing verification |
| 2 | ✅ Complete | Methodology drafting, dry runs |
| 3 | ✅ Complete | Corruption modules, experimental design |
| **4** | **🔄 In Progress** | **Pilot experiments and validation** |
| 5 | ⏳ Pending | Stability and bias-variance analysis |
| 6 | ⏳ Pending | Tabular domain experiments |
| 7 | ⏳ Pending | Text domain experiments |
| 8 | ⏳ Pending | Regression robustness, full analysis |
| 9 | ⏳ Pending | Discussion and final revisions |
| 10 | ⏳ Pending | Final submission |

## Key Files

- **Main Guide:** `README.md`
- **Week 4 Guide:** `docs/WEEK4_PILOT_GUIDE.md`
- **Week 4 Checklist:** `docs/WEEK4_CHECKLIST.md`
- **Documentation Index:** `docs/DOCUMENTATION_INDEX.md`
- **Project Plan:** `../Plans/Winter-Term-Revised-Plan.md`

## Quick Commands

**Run a pilot:**
```bash
python -m src.cli.run_severity_grid --config configs/adult_noise.yaml --n-points 11
```

**Plot results:**
```bash
python scripts/plot_results_simple.py --dir outputs/severity_grids
```

**Check status:**
```bash
ls outputs/severity_grids/
```
