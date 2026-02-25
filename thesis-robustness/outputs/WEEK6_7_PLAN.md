# Weeks 6 & 7: Execution Plan

**Reference:** Winter Term Revised Plan  
**Week 6:** Tabular domain experiments (Adult Income)  
**Week 7:** Text domain experiments (IMDB → Amazon)

---

## Week 6 Deliverables (from plan)

| Deliverable | Description |
|-------------|-------------|
| Aggregated tabular results | Full robustness experiments on Adult: noise, missingness, imbalance × RF, XGB, SVM |
| Draft degradation curves | Plots and tables for each (corruption, model) pair with multi-seed aggregates |

### Week 6: Experiment Matrix

| Corruption | Severity range | Points | Models | Seeds |
|------------|----------------|--------|--------|-------|
| Additive noise | 0.0 → 1.0 | 11 | RF, XGB, SVM-RBF | 42, 43, 44 |
| Missingness | 0.0 → 0.5 | 6 | RF, XGB, SVM-RBF | 42, 43, 44 |
| Class imbalance | 0.1 → 1.0 | 5 (e.g. 0.1, 0.25, 0.5, 0.75, 1.0) | RF, XGB, SVM-RBF | 42, 43, 44 |

**Output directories:** `outputs/week6/adult_noise_{model}/`, `outputs/week6/adult_missingness_{model}/`, `outputs/week6/adult_imbalance_{model}/`

### Week 6: Run commands (from repo root `thesis-robustness`)

```bash
# Additive noise — full grid, 3 seeds, 3 models
python -m src.cli.run_severity_grid --config configs/adult_noise.yaml \
  --output-dir outputs/week6/adult_noise_rf --seeds 42,43,44 --model random_forest
python -m src.cli.run_severity_grid --config configs/adult_noise.yaml \
  --output-dir outputs/week6/adult_noise_xgb --seeds 42,43,44 --model xgboost
python -m src.cli.run_severity_grid --config configs/adult_noise.yaml \
  --output-dir outputs/week6/adult_noise_svm --seeds 42,43,44 --model svm_rbf

# Missingness — 0 to 0.5, 6 points
python -m src.cli.run_severity_grid --config configs/adult_missingness.yaml \
  --min-severity 0 --max-severity 0.5 --n-points 6 \
  --output-dir outputs/week6/adult_missingness_rf --seeds 42,43,44 --model random_forest
python -m src.cli.run_severity_grid --config configs/adult_missingness.yaml \
  --min-severity 0 --max-severity 0.5 --n-points 6 \
  --output-dir outputs/week6/adult_missingness_xgb --seeds 42,43,44 --model xgboost
python -m src.cli.run_severity_grid --config configs/adult_missingness.yaml \
  --min-severity 0 --max-severity 0.5 --n-points 6 \
  --output-dir outputs/week6/adult_missingness_svm --seeds 42,43,44 --model svm_rbf

# Class imbalance — severities 0.1, 0.25, 0.5, 0.75, 1.0
python -m src.cli.run_severity_grid --config configs/adult_imbalance.yaml \
  --severities 0.1,0.25,0.5,0.75,1.0 \
  --output-dir outputs/week6/adult_imbalance_rf --seeds 42,43,44 --model random_forest
python -m src.cli.run_severity_grid --config configs/adult_imbalance.yaml \
  --severities 0.1,0.25,0.5,0.75,1.0 \
  --output-dir outputs/week6/adult_imbalance_xgb --seeds 42,43,44 --model xgboost
python -m src.cli.run_severity_grid --config configs/adult_imbalance.yaml \
  --severities 0.1,0.25,0.5,0.75,1.0 \
  --output-dir outputs/week6/adult_imbalance_svm --seeds 42,43,44 --model svm_rbf
```

---

## Week 7 Deliverables (from plan)

| Deliverable | Description |
|-------------|-------------|
| Text robustness results | IMDB token dropout × models; degradation curves |
| Model comparison summary | SVM (linear), Random Forest, XGBoost on IMDB |
| IMDB → Amazon domain shift | Train on IMDB only; evaluate on IMDB test and on Amazon (out-of-domain) |

### Week 7: Experiment Matrix

| Experiment | Description | Models | Seeds |
|------------|-------------|--------|-------|
| IMDB token dropout | Severity 0 → 0.5, 6 points | linear_svm, random_forest, xgboost | 42, 43, 44 |
| IMDB → Amazon domain shift | No severity; train IMDB, eval IMDB test + Amazon | linear_svm, random_forest, xgboost | 42, 43, 44 |

**Output directories:**  
- Token dropout: `outputs/week7/imdb_token_dropout_{model}/`  
- Domain shift: `outputs/week7/imdb_to_amazon_{model}/` (from domain-shift script)

### Week 7: Run commands

```bash
# Token dropout — 6 points, 3 models
python -m src.cli.run_severity_grid --config configs/imdb_token_dropout.yaml \
  --min-severity 0 --max-severity 0.5 --n-points 6 \
  --output-dir outputs/week7/imdb_token_dropout_linear_svm --seeds 42,43,44 --model linear_svm
python -m src.cli.run_severity_grid --config configs/imdb_token_dropout.yaml \
  --min-severity 0 --max-severity 0.5 --n-points 6 \
  --output-dir outputs/week7/imdb_token_dropout_rf --seeds 42,43,44 --model random_forest
python -m src.cli.run_severity_grid --config configs/imdb_token_dropout.yaml \
  --min-severity 0 --max-severity 0.5 --n-points 6 \
  --output-dir outputs/week7/imdb_token_dropout_xgb --seeds 42,43,44 --model xgboost

# Domain shift (IMDB train → IMDB test and Amazon) — use dedicated script
python scripts/run_domain_shift.py --seeds 42,43,44 --output-dir outputs/week7/imdb_to_amazon
```

---

## Summary

- **Week 6:** 9 severity grids (3 corruptions × 3 models), each with 3 seeds. Results → `WEEK6_RESULTS.md` and plots in `outputs/week6/`.
- **Week 7:** 3 token-dropout grids + 1 domain-shift run (3 models × 3 seeds). Results → `WEEK7_RESULTS.md` and plots in `outputs/week7/`.

Run `scripts/run_week6_week7.sh` to execute all commands (or run the Python domain-shift script separately if data paths differ).
