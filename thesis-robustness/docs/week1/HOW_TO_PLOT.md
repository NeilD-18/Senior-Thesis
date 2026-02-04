# How to Create Graphs from Your Results

## Quick Method (Recommended)

Run this command to generate degradation curves:

```bash
cd thesis-robustness
source venv/bin/activate
python scripts/plot_results_simple.py \
    --dir outputs/severity_grids \
    --plot outputs/severity_grids/degradation_curves.png
```

**What this does:**
- Reads results from all your severity grid experiments
- Creates a 3-panel plot showing Accuracy, F1-Score, and AUROC vs Severity
- Saves the plot as `degradation_curves.png`

**Output:** You'll get a plot file and a summary table printed to console.

## What You'll See

The plot will show:
- **X-axis**: Corruption severity (0.0 = clean, 1.0 = maximum corruption)
- **Y-axis**: Performance metric (Accuracy/F1/AUROC)
- **Blue line**: Test set performance
- **Orange line**: Validation set performance

## Example Output

```
RESULTS SUMMARY
======================================================================
Severity    Test Acc     Test F1      Test AUROC  
--------------------------------------------------
0.00        0.8548       0.8292       0.8789
0.10        0.8531       0.8232       0.8756
0.20        0.8507       0.8186       0.8743
...
1.00        0.8371       0.7705       0.8309

✓ Saved plot to outputs/severity_grids/degradation_curves.png
✓ Done!
```

## Troubleshooting

**If the script doesn't run:**
1. Make sure virtual environment is activated: `source venv/bin/activate`
2. Check that matplotlib is installed: `pip install matplotlib`
3. Verify results exist: `ls outputs/severity_grids/`

**If plot is empty:**
- Check that `final_metrics.json` files exist in each run directory
- Verify severity values are being extracted (check directory names)

## Alternative: Use the Analysis Script

If the simple script doesn't work, try:

```bash
python -m src.cli.analyze_severity_grid \
    --summary outputs/severity_grids/severity_grid_summary.yaml \
    --plot outputs/severity_grids/degradation_curves.png
```

## Viewing Your Plot

After generation, open the plot:
```bash
# On Mac
open outputs/severity_grids/degradation_curves.png

# On Linux
xdg-open outputs/severity_grids/degradation_curves.png
```

## For Your Thesis

The generated plot shows:
1. **How performance degrades** as corruption increases
2. **Model robustness** - gradual decline = robust, sharp drop = sensitive
3. **Comparison** between validation and test performance

Use this plot in your Results section to show degradation curves!
