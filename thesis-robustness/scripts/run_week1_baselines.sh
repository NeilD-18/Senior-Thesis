#!/bin/bash
# Script to run all Week 1 baseline experiments

set -e  # Exit on error

# Get the directory where the script is located and go to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists and isn't already activated
if [ -d "venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Use python from venv if available, otherwise use python3
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
else
    PYTHON_CMD="python3"
fi

echo "========================================="
echo "Week 1 Baseline Experiments"
echo "========================================="
echo ""

# Check if dependencies are installed
echo "Checking dependencies..."
$PYTHON_CMD -c "import pandas, sklearn, xgboost, yaml, ucimlrepo" 2>/dev/null || {
    echo "ERROR: Dependencies not installed. Please run:"
    echo "  source venv/bin/activate"
    echo "  pip install -e ."
    echo "  or"
    echo "  pip install numpy pandas scikit-learn xgboost pyyaml ucimlrepo"
    exit 1
}
echo "✓ Dependencies OK"
echo ""

# Create output directories
mkdir -p outputs/runs outputs/summary

# Run baseline experiments
echo "Running baseline experiments..."
echo ""

echo "1. Adult Income dataset..."
$PYTHON_CMD -m src.cli.run_baseline --config configs/adult_baseline.yaml --run-name adult_baseline || {
    echo "WARNING: Adult baseline failed (may need to download dataset)"
}

echo ""
echo "2. IMDB dataset..."
$PYTHON_CMD -m src.cli.run_baseline --config configs/imdb_baseline.yaml --run-name imdb_baseline || {
    echo "WARNING: IMDB baseline failed"
}

echo ""
echo "3. Amazon Reviews dataset..."
$PYTHON_CMD -m src.cli.run_baseline --config configs/amazon_baseline.yaml --run-name amazon_baseline || {
    echo "WARNING: Amazon baseline failed"
}

echo ""
echo "4. Airbnb dataset..."
$PYTHON_CMD -m src.cli.run_baseline --config configs/airbnb_baseline.yaml --run-name airbnb_baseline || {
    echo "WARNING: Airbnb baseline failed (may need to download dataset)"
}

echo ""
echo "========================================="
echo "Generating summary..."
echo "========================================="
$PYTHON_CMD -m src.cli.summarize --output outputs/summary/baseline_results.csv

echo ""
echo "========================================="
echo "Week 1 baselines complete!"
echo "Results saved to: outputs/summary/baseline_results.csv"
echo "========================================="
