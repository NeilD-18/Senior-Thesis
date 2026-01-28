# Visualization Guide

How to create graphs and visualizations from your experiment results.

## Quick Plot Generation

The easiest way to create degradation curves:

```bash
python scripts/plot_results_simple.py \
    --dir outputs/severity_grids \
    --plot outputs/severity_grids/degradation_curves.png
```

This will:
1. Load results from all run directories
2. Extract severity values from directory names
3. Create a 3-panel plot (Accuracy, F1-Score, AUROC)
4. Save to the specified path

## What the Plot Shows

The degradation curve plot contains three subplots:

1. **Accuracy vs Severity**: Overall correctness as corruption increases
2. **F1-Score vs Severity**: Balanced precision/recall as corruption increases
3. **AUROC vs Severity**: Class discrimination ability as corruption increases

Each plot shows:
- **Blue line (Test)**: Performance on test set
- **Orange line (Validation)**: Performance on validation set
- **X-axis**: Corruption severity (0.0 = clean, 1.0 = maximum)
- **Y-axis**: Metric value

## Interpreting the Curves

### Robust Model (Good)
```
Performance
    |
0.9 |●
    |  ●
0.8 |    ●
    |      ●
0.7 |        ●
    |___________
    0.0  0.5  1.0  Severity
```
- Gradual, smooth decline
- Maintains >80% performance even at high severity

### Sensitive Model (Bad)
```
Performance
    |
0.9 |●
    |
0.8 |
    |
0.7 |
    |        ●
0.6 |          ●
    |___________
    0.0  0.5  1.0  Severity
```
- Sharp drop at low severities
- Performance collapses quickly

## Customizing Plots

### Change Output Location
```bash
python scripts/plot_results_simple.py \
    --dir outputs/severity_grids \
    --plot my_custom_plot.png
```

### Plot Specific Directory
```bash
python scripts/plot_results_simple.py \
    --dir outputs/my_experiment \
    --plot outputs/my_experiment/results.png
```

## Advanced Visualization

### Using Python Directly

```python
import json
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Load results
results = []
for run_dir in Path('outputs/severity_grids').iterdir():
    if not run_dir.is_dir():
        continue
    
    metrics_file = run_dir / 'final_metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Extract severity from directory name
        match = re.search(r'_(\d+\.?\d*)_', run_dir.name)
        if match:
            severity = float(match.group(1))
            results.append({
                'severity': severity,
                'accuracy': metrics.get('test_accuracy')
            })

# Sort and plot
results.sort(key=lambda x: x['severity'])
severities = [r['severity'] for r in results]
accuracies = [r['accuracy'] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(severities, accuracies, 'o-', linewidth=2, markersize=8)
plt.xlabel('Corruption Severity')
plt.ylabel('Test Accuracy')
plt.title('Performance Degradation')
plt.grid(True, alpha=0.3)
plt.savefig('my_plot.png', dpi=150)
```

## Plotting Multiple Experiments

To compare different models or corruptions:

```python
import matplotlib.pyplot as plt
from scripts.plot_results_simple import load_results_from_dirs

# Load results from different experiments
rf_results = load_results_from_dirs('outputs/severity_grids_rf')
xgb_results = load_results_from_dirs('outputs/severity_grids_xgb')

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot([r['severity'] for r in rf_results], 
         [r['test_accuracy'] for r in rf_results], 
         'o-', label='Random Forest', linewidth=2)
plt.plot([r['severity'] for r in xgb_results], 
         [r['test_accuracy'] for r in xgb_results], 
         's-', label='XGBoost', linewidth=2)
plt.xlabel('Corruption Severity')
plt.ylabel('Test Accuracy')
plt.title('Model Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('comparison.png', dpi=150)
```

## Troubleshooting

**Problem**: Plot not generating
- Check that run directories exist in the specified path
- Verify `final_metrics.json` files exist in each run directory
- Make sure matplotlib is installed: `pip install matplotlib`

**Problem**: Missing data points
- Some runs may have failed - check console output
- Verify severity values are being extracted correctly from directory names

**Problem**: Plot looks wrong
- Check that severity values are sorted correctly
- Verify metrics are being read from the correct keys

## Output File Format

The plot is saved as PNG with:
- **Resolution**: 150 DPI (suitable for presentations)
- **Size**: 15x5 inches (3 subplots side-by-side)
- **Format**: PNG (can be converted to PDF/PNG for papers)

## Next Steps

After generating plots:
1. Review the degradation patterns
2. Compare with baseline (severity=0.0)
3. Calculate robustness statistics (see RESULTS_GUIDE.md)
4. Use plots in thesis/presentations
