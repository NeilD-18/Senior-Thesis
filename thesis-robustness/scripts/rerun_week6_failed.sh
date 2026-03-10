#!/usr/bin/env bash
# Re-run only Week 6 grids that failed due transient Adult fetch errors.
set -e
cd "$(dirname "$0")/.."

echo "=== Week 6 recovery rerun (failed grids only) ==="

python -m src.cli.run_severity_grid --config configs/adult_noise.yaml \
  --output-dir outputs/week6_rerun/adult_noise_svm --seeds 42,43,44 --model svm_rbf

python -m src.cli.run_severity_grid --config configs/adult_missingness.yaml \
  --min-severity 0 --max-severity 0.5 --n-points 6 \
  --output-dir outputs/week6_rerun/adult_missingness_rf --seeds 42,43,44 --model random_forest
python -m src.cli.run_severity_grid --config configs/adult_missingness.yaml \
  --min-severity 0 --max-severity 0.5 --n-points 6 \
  --output-dir outputs/week6_rerun/adult_missingness_xgb --seeds 42,43,44 --model xgboost
python -m src.cli.run_severity_grid --config configs/adult_missingness.yaml \
  --min-severity 0 --max-severity 0.5 --n-points 6 \
  --output-dir outputs/week6_rerun/adult_missingness_svm --seeds 42,43,44 --model svm_rbf

python -m src.cli.run_severity_grid --config configs/adult_imbalance.yaml \
  --severities 0.02,0.05,0.1,0.15,0.199 \
  --output-dir outputs/week6_rerun/adult_imbalance_rf --seeds 42,43,44 --model random_forest
python -m src.cli.run_severity_grid --config configs/adult_imbalance.yaml \
  --severities 0.02,0.05,0.1,0.15,0.199 \
  --output-dir outputs/week6_rerun/adult_imbalance_xgb --seeds 42,43,44 --model xgboost
python -m src.cli.run_severity_grid --config configs/adult_imbalance.yaml \
  --severities 0.02,0.05,0.1,0.15,0.199 \
  --output-dir outputs/week6_rerun/adult_imbalance_svm --seeds 42,43,44 --model svm_rbf

echo "=== Week 6 recovery rerun complete ==="
