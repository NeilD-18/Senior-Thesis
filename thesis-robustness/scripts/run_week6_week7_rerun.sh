#!/usr/bin/env bash
# Re-run Week 6 and Week 7 experiments into fresh output directories.
# Execute from repo root: thesis-robustness/
set -e
cd "$(dirname "$0")/.."

echo "=== Week 6 rerun: Adult tabular experiments ==="

# Additive noise — 11 points, 3 seeds, 3 models
python -m src.cli.run_severity_grid --config configs/adult_noise.yaml \
  --output-dir outputs/week6_rerun/adult_noise_rf --seeds 42,43,44 --model random_forest
python -m src.cli.run_severity_grid --config configs/adult_noise.yaml \
  --output-dir outputs/week6_rerun/adult_noise_xgb --seeds 42,43,44 --model xgboost
python -m src.cli.run_severity_grid --config configs/adult_noise.yaml \
  --output-dir outputs/week6_rerun/adult_noise_svm --seeds 42,43,44 --model svm_rbf

# Missingness — 0 to 0.5, 6 points
python -m src.cli.run_severity_grid --config configs/adult_missingness.yaml \
  --min-severity 0 --max-severity 0.5 --n-points 6 \
  --output-dir outputs/week6_rerun/adult_missingness_rf --seeds 42,43,44 --model random_forest
python -m src.cli.run_severity_grid --config configs/adult_missingness.yaml \
  --min-severity 0 --max-severity 0.5 --n-points 6 \
  --output-dir outputs/week6_rerun/adult_missingness_xgb --seeds 42,43,44 --model xgboost
python -m src.cli.run_severity_grid --config configs/adult_missingness.yaml \
  --min-severity 0 --max-severity 0.5 --n-points 6 \
  --output-dir outputs/week6_rerun/adult_missingness_svm --seeds 42,43,44 --model svm_rbf

# Class imbalance — corrected severity levels near Adult natural ratio
python -m src.cli.run_severity_grid --config configs/adult_imbalance.yaml \
  --severities 0.02,0.05,0.1,0.15,0.199 \
  --output-dir outputs/week6_rerun/adult_imbalance_rf --seeds 42,43,44 --model random_forest
python -m src.cli.run_severity_grid --config configs/adult_imbalance.yaml \
  --severities 0.02,0.05,0.1,0.15,0.199 \
  --output-dir outputs/week6_rerun/adult_imbalance_xgb --seeds 42,43,44 --model xgboost
python -m src.cli.run_severity_grid --config configs/adult_imbalance.yaml \
  --severities 0.02,0.05,0.1,0.15,0.199 \
  --output-dir outputs/week6_rerun/adult_imbalance_svm --seeds 42,43,44 --model svm_rbf

echo "=== Week 7 rerun: IMDB token dropout ==="
python -m src.cli.run_severity_grid --config configs/imdb_token_dropout.yaml \
  --min-severity 0 --max-severity 0.5 --n-points 6 \
  --output-dir outputs/week7_rerun/imdb_token_dropout_linear_svm --seeds 42,43,44 --model linear_svm
python -m src.cli.run_severity_grid --config configs/imdb_token_dropout.yaml \
  --min-severity 0 --max-severity 0.5 --n-points 6 \
  --output-dir outputs/week7_rerun/imdb_token_dropout_rf --seeds 42,43,44 --model random_forest
python -m src.cli.run_severity_grid --config configs/imdb_token_dropout.yaml \
  --min-severity 0 --max-severity 0.5 --n-points 6 \
  --output-dir outputs/week7_rerun/imdb_token_dropout_xgb --seeds 42,43,44 --model xgboost

echo "=== Week 7 rerun: IMDB -> Amazon domain shift ==="
python scripts/run_domain_shift.py --seeds 42,43,44 --output-dir outputs/week7_rerun/imdb_to_amazon
python scripts/plot_week7_summary.py --week7-dir outputs/week7_rerun

echo "=== Week 6 & 7 rerun complete. See outputs/week6_rerun/ and outputs/week7_rerun/ ==="
