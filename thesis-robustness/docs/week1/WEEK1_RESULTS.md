# Week 1 Baseline Results

## Classification Results

| Dataset | Model | Samples | Test Accuracy | Test F1 | Test AUROC | Status |
|---------|-------|---------|---------------|---------|------------|--------|
| **Adult Income** | Random Forest | 45,222 | 85.48% | 82.92% | 87.89% | ✅ |
| **IMDB** | Linear SVM | 25,000 | 85.84% | 85.84% | 93.66% | ✅ |
| **Amazon Books** | Linear SVM | 2,000 | 79.50% | 79.48% | 88.11% | ✅ |

## Regression Results

| Dataset | Model | Samples | Test RMSE | Test MAE | Status |
|---------|-------|---------|-----------|----------|--------|
| **Airbnb** | Random Forest | 54,110 | 0.409 | 0.296 | ✅ |

## Analysis

### Adult Income (Tabular Classification)
- **Class distribution:** Highly imbalanced (83.4% class 0, 16.6% class 1)
- **Accuracy vs F1:** F1 is lower (82.92% vs 85.48%) — expected for imbalanced data
- **AUROC:** 87.89% indicates good class separation
- **Verdict:** ✅ Results are reasonable

### IMDB (Text Classification)
- **Class distribution:** Perfectly balanced (50/50)
- **Accuracy == F1:** Expected for balanced datasets
- **AUROC:** 93.66% — excellent discrimination
- **Dataset size:** 25,000 reviews (12,500 pos / 12,500 neg)
- **Verdict:** ✅ Results are reasonable

### Amazon Books (Text Classification)
- **Class distribution:** Perfectly balanced (50/50)
- **Accuracy == F1:** Expected for balanced datasets
- **AUROC:** 88.11% — good discrimination
- **Performance:** Lower than IMDB (79.5% vs 85.8%)
- **Dataset size:** 2,000 reviews (smaller than IMDB)
- **Verdict:** ✅ Reasonable, but performance gap needs investigation

### Airbnb (Regression)
- **Target:** Log-transformed price (range: 1.79-7.60)
- **RMSE:** 0.409 on log scale
- **MAE:** 0.296 on log scale
- **In original scale:** exp(0.409) ≈ 1.51x price error
- **Verdict:** ✅ Reasonable for baseline

## Potential Concerns

### 1. Amazon Performance Lower Than IMDB (79.5% vs 85.8%)

**Possible Reasons:**
1. **Smaller dataset:** 2,000 samples vs 25,000 samples
2. **Domain difference:** Product reviews vs movie reviews
3. **Trained on different data:** Both trained on their own datasets, not cross-domain yet

**To Investigate:**
- Train on IMDB, test on Amazon (true cross-domain evaluation)
- Compare vocabulary overlap between datasets
- Check if 2,000 samples is enough for good generalization

### 2. F1 == Accuracy for Balanced Datasets

**Status:** ✅ This is mathematically correct

For perfectly balanced datasets:
- Weighted F1 = (F1_class0 × 0.5) + (F1_class1 × 0.5)
- When precision ≈ recall for both classes, weighted F1 ≈ accuracy

### 3. Missing AUROC (Now Fixed)

**Was:** LinearSVC doesn't have `predict_proba()`  
**Fix:** Use `decision_function()` for AUROC calculation  
**Status:** ✅ Fixed - All models now report AUROC

## Comparison to Literature

| Dataset | Metric | Our Result | Expected Range | Status |
|---------|--------|------------|----------------|--------|
| Adult | Accuracy | 85.48% | 82-87% (RF) | ✅ Within range |
| IMDB | Accuracy | 85.84% | 85-90% (SVM+TF-IDF) | ✅ Within range |
| IMDB | AUROC | 93.66% | 90-95% | ✅ Good |
| Amazon | Accuracy | 79.50% | 75-82% (in-domain) | ✅ Within range |
| Airbnb | RMSE | 0.409 | 0.3-0.5 (log scale) | ✅ Within range |

## Week 1 Deliverables - Status

- [x] Run baseline models on Adult, IMDB, and Airbnb datasets ✅
- [x] Verify preprocessing pipelines ✅
- [x] Ensure reproducibility (seed management) ✅
- [x] Generate baseline accuracy/F1/RMSE tables ✅
- [x] Logging system working ✅

## Next Steps (Week 2)

1. **Cross-domain evaluation:** Train IMDB model, test on Amazon
2. **Methodology section:** Document preprocessing, models, metrics
3. **Preliminary plots:** Visualize baseline results
4. **Additional models:** Test other algorithms (Logistic, SVM-RBF)

## Notes

- All experiments use seed=42 for reproducibility
- Train/Val/Test split: 70/10/20
- TF-IDF parameters: max_features=5000, ngram_range=(1,2)
- Text data kept as sparse matrices for efficiency
- Adult dataset auto-downloads from UCI
- IMDB and Amazon datasets use local files
- Airbnb dataset downloaded from Kaggle
