# Week 1 Baseline Results Verification

## Summary of Results

| Dataset | Model | Test Acc | Test F1 | Test AUROC | Notes |
|---------|-------|----------|---------|------------|-------|
| Adult | Random Forest | 85.48% | 82.92% | 87.89% | ✅ Correct |
| IMDB | Linear SVM | 85.84% | 85.84% | N/A | ⚠️ Missing AUROC |
| Amazon | Linear SVM | 79.50% | 79.48% | N/A | ⚠️ Missing AUROC |
| Airbnb | Random Forest | 0.409 RMSE | 0.296 MAE | N/A | ✅ Correct |

## Issues Identified

### 1. ✅ Accuracy == F1 for IMDB and Amazon (CORRECT)
**Status:** Not a bug!

**Explanation:**
- IMDB: Perfectly balanced (12,500 pos / 12,500 neg)
- Amazon: Perfectly balanced (1,000 pos / 1,000 neg)
- For perfectly balanced datasets, weighted F1 ≈ accuracy

**Verification:**
```
IMDB: 50/50 split
Amazon Books: 50/50 split
```

### 2. ⚠️ Missing AUROC for Text Models
**Status:** Bug - needs fixing

**Problem:**
- LinearSVC doesn't have `predict_proba()` method
- Current code only calculates AUROC when `predict_proba()` exists
- Should use `decision_function()` instead

**Fix Required:**
Update baseline pipeline to use `decision_function()` for LinearSVC models.

### 3. ✅ Adult Dataset Results (CORRECT)
**Status:** Correct

**Class Distribution:**
- Class 0 (<=50K): 37,714 (83.4%)
- Class 1 (>50K): 7,508 (16.6%)
- Imbalance ratio: 5.02:1

**Why F1 < Accuracy:**
- Accuracy: 85.48% (may be inflated by majority class)
- F1: 82.92% (accounts for class imbalance)
- AUROC: 87.89% (good discrimination)

This is expected behavior for imbalanced datasets.

### 4. ✅ Amazon Lower Than IMDB (CORRECT - Potentially)
**Status:** Expected

**Possible Explanations:**
1. **Different vocabulary** - Amazon product reviews vs movie reviews
2. **Smaller dataset** - 2,000 samples vs 25,000 samples
3. **Domain difference** - Different writing styles

**To Verify:**
- Amazon is meant to test cross-domain robustness (IMDB → Amazon)
- Lower performance on Amazon is expected if trained on IMDB

## Recommendations

### High Priority Fixes

1. **Add AUROC for LinearSVC models**
   - Use `decision_function()` instead of `predict_proba()`
   - Update `baseline.py` to handle both cases

2. **Add per-class metrics for Adult**
   - Report precision/recall per class
   - Show confusion matrix
   - Helps understand if model is biased to majority class

### Medium Priority

3. **Cross-domain evaluation**
   - Train on IMDB, test on Amazon (true cross-domain test)
   - Currently both are trained on their own datasets

4. **Add baseline comparisons**
   - Majority class baseline for Adult
   - Random baseline
   - Helps contextualize results

## Expected Ranges (from literature)

- **Adult Income:** 82-87% accuracy (Random Forest) ✅
- **IMDB:** 85-90% accuracy (Linear SVM on TF-IDF) ✅
- **Amazon Books:** 75-82% accuracy (in-domain) ✅
- **Airbnb:** RMSE 0.3-0.5 (log-transformed price) ✅

## Conclusion

**Current Status:** Results are mostly correct, with one missing feature (AUROC for text models).

**Action Items:**
1. Fix LinearSVC AUROC calculation (use decision_function)
2. Verify cross-domain setup (train IMDB, test Amazon)
3. Add per-class metrics for imbalanced datasets

**Overall Assessment:** ✅ Results are reasonable and match expected baselines from literature.
