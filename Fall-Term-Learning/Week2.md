# Week 2 – Chapter Notes & Thesis Connections  
_Machine Learning (Lantz, 4th Edition) – Chapters 6 & 10_

---

## Chapter 6 – Regression Methods

### What is Regression?

- Regression models predict a **continuous target variable (y)** based on one or more predictors (x).  
- The goal is **quantitative prediction**, not classification.  
- Beyond prediction, regression can reveal **relationships and causal patterns** between variables.  
- Examples include predicting profit, hospital turnaround time, or wine quality scores.

---

### Correlation

- Measures the **linear relationship** between two variables (–1 ≤ r ≤ 1).  
- Useful for detecting associations before modeling.  
- Can be misleading when relationships are **nonlinear**.  
- **Correlation matrices** help visualize relationships and guide feature selection.

---

### Linear Regression

- A **parametric model** that finds coefficients \( \beta \) minimizing the **sum of squared residuals**.  
- Residual = actual – predicted value.  
- **Interpretation:** each coefficient represents the expected change in y for a one-unit increase in x, holding others constant.  
- “Linear” refers to linearity in parameters, not necessarily in variables — polynomial and interaction terms can model curvature.  

**Common evaluation metrics:**
- **RMSE (Root Mean Square Error)** → measures average error magnitude.  
- **MAE (Mean Absolute Error)** → less sensitive to outliers.  
- **R² / Adjusted R²** → proportion of variance in y explained by predictors.  
  - Adjusted R² penalizes adding unnecessary predictors.  

---

### Regression Residuals

- Residuals reflect the gap between observed and predicted values.  
- Analyzing residuals reveals:
  - Nonlinear trends not captured by the model.  
  - Heteroskedasticity (unequal variance).  
  - Outliers or missing data issues.  
- The **standard deviation of residuals (Se)** gives a direct sense of prediction accuracy.  
- In my project, I can use **residual behavior** to evaluate robustness when noise or missingness is introduced.

---

### Regression Trees

- Similar to classification trees, but used for **continuous y**.  
- The algorithm selects splits that **minimize variance** (standard deviation reduction).  
- Predictions for test data = **mean y** of the training samples in each terminal node.  
- Trees cannot predict beyond the range of values seen in training data.  

**Example:** Predicting wine quality using acidity, sugar, and alcohol levels.  

---

### Avoiding Overfitting

- CART models can overfit if grown too deep — each observation may become its own node.  
- **Pruning** reduces branches to improve generalization.  
- Holding out a **test set** or using **cross-validation** helps detect overfitting.  
- Dropping irrelevant features can improve tree stability since unnecessary splits can distort structure.  

---

### Advanced Regression Models

- **Random Forests** → use bagging to average multiple trees, reducing variance.  
- **XGBoost** → boosting method that fits residuals iteratively for improved accuracy.  
- **M5P (RWeka)** → fits regression models at tree leaves instead of using simple averages.  

These ensemble and hybrid methods tend to be **more robust** to noise, missingness, and small perturbations — a key focus in my thesis.  

---

### Chapter 6 Takeaway

> Regression shifts ML from classification to continuous prediction and introduces bias–variance tradeoffs central to robustness testing.  

In my project, regression models (like Random Forests and XGBoost) will help quantify how **prediction error increases** under corruption or distribution shift, using metrics like RMSE and R² degradation as robustness indicators.  

---

## Chapter 10 – Model Evaluation II

### Why Models Fail

- Many ML models fail in real-world deployment despite strong training accuracy.  
- This happens when they lack a **basis for inference** — i.e., confidence that results will generalize to future data.  
- Fitting one dataset well does **not** guarantee future reliability, especially under new conditions or different data pedigrees.  

---

### Improving Evaluation

- A single train/test split offers only one random snapshot of performance.  
- This can be **lucky or unlucky** depending on how the data happens to split.  
- More robust evaluation requires **repeated sampling** and **structured validation frameworks**.

---

### K-Fold Cross-Validation

- Data is divided into **k subsets (folds)**.  
- The model trains on (k–1) folds and tests on the remaining one.  
- This process repeats k times, each fold serving as the test set once.  
- Final accuracy = average across all folds.  
- Commonly, **k = 10**.  

This provides a more stable measure of out-of-sample performance and reduces variance from random splits.  

---

### Train / Test / Validate Framework

- **Training set:** used to build the model.  
- **Test set:** used to tune hyperparameters and compare models.  
- **Validation set:** used for the final, unbiased performance evaluation.  

This ensures that the test data doesn’t indirectly influence model selection — maintaining the independence of the validation step.  

---

### Data Pedigree and True Generalization

- Even k-fold CV can overestimate accuracy because all folds share the same **data pedigree** (collected under the same conditions).  
- For real robustness testing, validation data should differ in meaningful ways:
  - **Time-based shifts** (e.g., later samples).  
  - **Geographic shifts** (e.g., new regions).  
  - **Domain shifts** (e.g., IMDB vs. Amazon reviews).  

Predicting well on such *qualitatively different* data is what indicates **true robustness** — the central question of my thesis.  

---

### Chapter 10 Takeaway

> Reliable evaluation requires measuring performance on data that differ from the training conditions.  

In my project, I’ll mirror this by training on one domain and testing on another, or by injecting controlled noise. This simulates how models face real-world drift and helps identify which are **most resilient to change**.  

---

## Big Picture

| Chapter | Core Ideas | Connection to My Thesis |
|----------|-------------|--------------------------|
| **6 – Regression Methods** | Continuous prediction, residual analysis, bias–variance tradeoff | Introduces regression techniques I’ll use to measure quantitative degradation under noise and shift. |
| **10 – Model Evaluation II** | Cross-validation, validation sets, data pedigree | Establishes how to evaluate model robustness and generalization across distributions. |

---

## Final Reflection

Ch. 6 deepened my understanding of regression and how model accuracy can deteriorate as data quality worsens.  
Ch. 10 introduced the frameworks I’ll use to **validate robustness** — specifically, testing on data with a different pedigree.  

Together, these chapters move from modeling to **evaluating reliability**, bridging theory with my thesis experiments on robustness under noise, missingness, and distribution shift.  
