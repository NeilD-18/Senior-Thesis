# Results Interpretation Guide

This guide explains how to interpret experiment results from the robustness study.

## Table of Contents

- [Understanding Output Files](#understanding-output-files)
- [Interpreting Metrics](#interpreting-metrics)
- [Reading Degradation Curves](#reading-degradation-curves)
- [Robustness Statistics](#robustness-statistics)
- [Comparing Models](#comparing-models)
- [Example Analysis](#example-analysis)

## Understanding Output Files

### Single Experiment Output

Each experiment creates a directory:

```
outputs/runs/<dataset>_<model>_<corruption>_<severity>_<timestamp>/
├── config.json          # Full experiment configuration
├── final_metrics.json   # Final metrics (JSON format)
└── metrics.csv          # Metrics history (CSV format)
```

**config.json**: Contains all parameters used in the experiment
- Dataset, model, corruption settings
- Hyperparameters, random seed
- Data split configuration

**final_metrics.json**: Final performance metrics
```json
{
  "val_accuracy": 0.866,
  "val_f1": 0.845,
  "val_auroc": 0.877,
  "test_accuracy": 0.855,
  "test_f1": 0.829,
  "test_auroc": 0.879
}
```

**metrics.csv**: Historical metrics (useful for training curves)

### Severity Grid Output

Severity grid experiments create:

```
outputs/severity_grids/
├── <run_name_0.0>/      # Individual experiment results
├── <run_name_0.1>/
├── ...
└── severity_grid_summary.yaml  # Summary of all severities
```

**severity_grid_summary.yaml**: Aggregated results across all severities
- Contains metrics for each severity level
- Used for generating degradation curves

## Interpreting Metrics

### Classification Metrics

**Accuracy**:
- **Definition**: Fraction of correct predictions
- **Range**: [0, 1], higher is better
- **Interpretation**: 
  - 0.85 = 85% of predictions are correct
  - Can be misleading with imbalanced classes

**F1-Score**:
- **Definition**: Harmonic mean of precision and recall
- **Range**: [0, 1], higher is better
- **Interpretation**:
  - Better than accuracy for imbalanced classes
  - 0.83 = good balance of precision and recall
  - Emphasized when class prevalence is altered

**AUROC** (Area Under ROC Curve):
- **Definition**: Ability to distinguish between classes
- **Range**: [0, 1], higher is better
- **Interpretation**:
  - 0.5 = random guessing
  - 0.7-0.8 = acceptable
  - 0.8-0.9 = excellent
  - >0.9 = outstanding

### Regression Metrics

**RMSE** (Root Mean Squared Error):
- **Definition**: Square root of average squared errors
- **Range**: [0, ∞), lower is better
- **Interpretation**: Penalizes large errors more
- **Units**: Same as target variable

**MAE** (Mean Absolute Error):
- **Definition**: Average absolute errors
- **Range**: [0, ∞), lower is better
- **Interpretation**: Equal weight to all errors
- **Units**: Same as target variable

## Reading Degradation Curves

A degradation curve plots performance vs corruption severity.

### Curve Shapes

**Gradual Decline** (Robust):
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
- Model maintains performance well
- Small, consistent degradation
- **Interpretation**: Model is robust to this corruption

**Sharp Drop** (Sensitive):
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
- Performance drops quickly
- Large degradation at low severities
- **Interpretation**: Model is sensitive to this corruption

**Plateau Then Drop**:
```
Performance
    |
0.9 |●
    |  ●●●
0.8 |      ●
    |        ●
0.7 |          ●
    |___________
    0.0  0.5  1.0  Severity
```
- Maintains performance initially
- Sharp drop at higher severities
- **Interpretation**: Tolerates low corruption, fails at high levels

### Key Points on Curve

1. **Clean Performance** (severity=0.0):
   - Baseline performance without corruption
   - Use for comparison

2. **Performance at Moderate Severity** (severity=0.5):
   - How well model handles moderate corruption
   - Practical robustness indicator

3. **Worst Performance** (severity=1.0):
   - Performance under maximum corruption
   - Worst-case scenario

## Robustness Statistics

When analyzing severity grids, compute these statistics:

### 1. Clean Performance
- Performance at severity=0.0
- Baseline for comparison

### 2. Maximum Drop
- Difference between clean and worst performance
- `max_drop = clean_performance - worst_performance`

### 3. Relative Drop
- Percentage decrease from clean to worst
- `relative_drop = (max_drop / clean_performance) * 100`
- Lower relative drop = more robust

### 4. Average Performance (AUC)
- Area under the degradation curve
- Average performance across all severities
- Higher AUC = more robust

### 5. Degradation Slope
- Rate of performance decline
- Linear fit slope (negative = degradation)
- Less negative = more robust

## Comparing Models

### Same Corruption, Different Models

Compare how different models handle the same corruption:

```
Model A (RF):  Clean=0.85, Worst=0.80, Drop=0.05 (5.9%)
Model B (XGB): Clean=0.87, Worst=0.75, Drop=0.12 (13.8%)
```

**Interpretation**: 
- Model A is more robust (smaller drop)
- Model B has higher clean performance but degrades more

### Same Model, Different Corruptions

Compare how the same model handles different corruptions:

```
Noise:        Clean=0.85, Worst=0.80, Drop=0.05 (5.9%)
Missingness:  Clean=0.85, Worst=0.70, Drop=0.15 (17.6%)
```

**Interpretation**:
- Model is more robust to noise than missingness
- Missingness causes larger degradation

## Example Analysis

### Example: Adult Dataset with Additive Noise

From your severity grid results:

```
Severity  Accuracy  F1-Score  AUROC
0.0       0.8548    0.8292    0.8789
0.5       0.8456    0.7980    0.8622
1.0       0.8371    0.7705    0.8309
```

**Analysis**:

1. **Clean Performance**: 85.48% accuracy (good baseline)

2. **Degradation**:
   - At severity 0.5: 84.56% (1.1% drop)
   - At severity 1.0: 83.71% (2.1% total drop)
   - **Relative drop**: 2.1% / 85.48% = 2.5%

3. **Robustness Assessment**:
   - Small, gradual degradation
   - Model maintains >83% accuracy even at maximum corruption
   - **Conclusion**: Random Forest is robust to additive noise on Adult dataset

4. **Comparison Points**:
   - F1 drops more than accuracy (0.829 → 0.771, 7% drop)
   - AUROC also degrades (0.879 → 0.831, 5.5% drop)
   - Suggests noise affects class separation more than overall accuracy

### Generating Analysis

Use the analysis script:

```bash
python -m src.cli.analyze_severity_grid \
    --summary outputs/severity_grids/severity_grid_summary.yaml \
    --plot outputs/severity_grids/degradation_curves.png
```

This generates:
- Summary table with all metrics
- Robustness statistics (AUC, drop, slope)
- Degradation curve plots
- Interpretation guide

## Best Practices

1. **Always compare to baseline**: Use severity=0.0 as reference
2. **Look at multiple metrics**: Accuracy, F1, AUROC tell different stories
3. **Consider practical severity**: Real-world corruption is often <0.5
4. **Statistical validation**: Run multiple seeds for confidence intervals
5. **Visual inspection**: Plot degradation curves to see patterns
6. **Context matters**: 2% drop might be significant or negligible depending on application

## Common Questions

**Q: What's a "good" robustness score?**
A: Depends on application. Generally:
- <5% relative drop: Very robust
- 5-15% relative drop: Moderately robust
- >15% relative drop: Sensitive

**Q: Should I focus on accuracy or F1?**
A: Use F1 when classes are imbalanced or when corruption affects class balance. Use accuracy for balanced problems.

**Q: How do I know if degradation is significant?**
A: Run multiple seeds and compute confidence intervals. If intervals don't overlap, difference is significant.

**Q: What severity range should I test?**
A: Typically 0.0 to 1.0 with 10-20 points. Focus on 0.0-0.5 for practical robustness.

## Next Steps

After analyzing results:

1. **Compare models**: Run same corruption on different models
2. **Compare corruptions**: Run different corruptions on same model
3. **Statistical validation**: Run multiple seeds for uncertainty quantification
4. **Visualization**: Create publication-quality degradation curves
5. **Documentation**: Record findings and interpretations
