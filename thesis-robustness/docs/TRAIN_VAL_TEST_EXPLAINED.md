# Train, Validation, and Test Sets Explained

Understanding the three-way data split and why it matters for your robustness study.

## The Three Sets

Your experiments split data into three parts:

```
Total Dataset
├── Training Set (70%)    ← Model learns from this
├── Validation Set (10%)  ← Used during development/tuning
└── Test Set (20%)        ← Final evaluation (never touched during training)
```

## What Each Set Does

### 1. **Training Set** (70% of data)

**Purpose**: Model learns patterns from this data

**What happens here:**
- Model sees the features and labels
- Model adjusts its parameters to fit the data
- **Corruption is applied here** (in your robustness experiments)

**In your experiments:**
- Corruption (noise, missingness, etc.) is added to training data
- Model learns to handle corrupted data
- This tests if the model can learn robustly despite corruption

**Example:**
```
Training data: 31,654 samples (Adult dataset)
- Corruption applied: Additive noise with severity 0.3
- Model trains on: Corrupted training data
```

### 2. **Validation Set** (10% of data)

**Purpose**: Check model performance during development

**What happens here:**
- Model makes predictions (but doesn't learn from it)
- Used to monitor training progress
- Can be used for hyperparameter tuning (though you're not doing this)
- **No corruption applied** - clean data

**Why it exists:**
- Gives you feedback during development
- Helps detect overfitting
- Provides a "sanity check" before final evaluation

**In your experiments:**
- Model predicts on clean validation data
- Shows how well model generalizes to clean data
- Useful for comparing across different corruption severities

**Example:**
```
Validation data: 4,523 samples (clean, no corruption)
- Model predicts on: Clean validation data
- Result: val_accuracy = 0.866 (86.6% correct)
```

### 3. **Test Set** (20% of data)

**Purpose**: Final, unbiased evaluation

**What happens here:**
- Model makes predictions (never learned from this data)
- This is the "real" performance estimate
- **No corruption applied** - clean data
- **Never used during training** - completely independent

**Why it exists:**
- Provides unbiased estimate of true performance
- Simulates how model will perform on new, unseen data
- This is what you report in your thesis

**In your experiments:**
- Model predicts on clean test data
- Shows true generalization ability
- This is your primary result

**Example:**
```
Test data: 9,045 samples (clean, no corruption)
- Model predicts on: Clean test data
- Result: test_accuracy = 0.855 (85.5% correct)
```

## Why Both Validation and Test?

### The Problem Without Both

If you only had train/test:
- You might tune hyperparameters on test set
- Test set becomes "contaminated" - you've seen it
- Performance on test set becomes overestimated
- Not a true measure of generalization

### The Solution: Three-Way Split

**Validation Set:**
- Use for development, tuning, monitoring
- Can look at it many times
- Helps you understand model behavior

**Test Set:**
- Use only once for final evaluation
- Never look at it during development
- True measure of performance

## In Your Robustness Study

### What Gets Corrupted?

**Training Set**: ✅ Corruption applied
- Additive noise
- Missingness
- Class imbalance
- Token dropout

**Validation Set**: ❌ No corruption
- Always clean
- Used to monitor performance

**Test Set**: ❌ No corruption
- Always clean
- Final evaluation

### Why This Design?

Your research question is:
> "How does training on corrupted data affect model performance?"

To answer this:
1. **Train** on corrupted data (simulates real-world noisy training)
2. **Evaluate** on clean data (simulates deployment on clean data)

This tests: "Can the model learn robustly despite training corruption?"

## Understanding Your Results

### Example from Your Results

```
Severity 0.3 (30% noise):
- val_accuracy: 0.8589 (85.89%)
- test_accuracy: 0.8498 (84.98%)
```

**What this means:**
- Model trained on data with 30% noise
- On clean validation data: 85.89% accurate
- On clean test data: 84.98% accurate
- Small difference suggests consistent performance

### Validation vs Test Metrics

**Validation metrics:**
- Slightly higher (often)
- Used during development
- Can be used for comparison across runs

**Test metrics:**
- More reliable estimate
- True generalization performance
- **This is what you report in your thesis**

### Why They Might Differ

1. **Random variation**: Different samples, different results
2. **Sample size**: Test set is larger (more stable)
3. **Data distribution**: Slight differences between splits

**Small differences (<2%) are normal and expected.**

## Best Practices

### For Your Thesis

1. **Report test metrics** as primary results
   - These are the most reliable
   - True generalization estimate

2. **Use validation metrics** for:
   - Monitoring during experiments
   - Quick comparisons
   - Understanding trends

3. **Both are useful** for:
   - Degradation curves (show both lines)
   - Checking consistency
   - Detecting issues

### In Your Plots

When you see both lines:
- **Blue (Test)**: Primary result - report this
- **Orange (Validation)**: Secondary - shows consistency

If they're close together: ✅ Good - consistent performance
If they're far apart: ⚠️ Check for issues (overfitting, data leakage, etc.)

## Common Questions

**Q: Why is validation accuracy higher than test accuracy?**
A: Common and normal. Validation set might be easier, or just random variation. Small differences (<2-3%) are fine.

**Q: Which should I trust more?**
A: **Test set** - it's the final, unbiased evaluation. But both should be similar.

**Q: Why not corrupt validation/test too?**
A: Your research question is about training robustness. You want to know: "If I train on noisy data, can I still perform well on clean data?" This simulates real-world scenarios where training data is noisy but deployment data is clean.

**Q: Should I use validation or test for degradation curves?**
A: **Both!** Show both lines. Test is primary, but validation shows consistency and helps verify results.

## Summary

| Set | Purpose | Corruption? | When to Use |
|-----|---------|-------------|-------------|
| **Train** | Model learning | ✅ Yes | Training only |
| **Validation** | Development/monitoring | ❌ No | During experiments |
| **Test** | Final evaluation | ❌ No | **Report in thesis** |

**Key takeaway**: Test set gives you the most reliable performance estimate. Validation set helps you understand model behavior. Both are useful, but test metrics are what you report as your primary results.
