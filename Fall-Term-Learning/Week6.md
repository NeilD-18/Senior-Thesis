# Week 6 – Model Improvement: Bagging, Boosting, and Random Forests  

## Concept and Intuition  

After evaluating model performance, the next step is figuring out **how to improve it**.  
This chapter introduces **systematic model-improvement techniques**—tuning, bootstrapping, bagging, and boosting—that reduce error, increase stability, and enhance generalization.  
These methods are central to my thesis on **robustness**, because they explicitly target the trade-off between bias and variance that governs how models behave under **distributional shifts**.

---

## Automated Tuning  

Model performance often depends on **hyperparameters** (e.g., *k* in k-NN, hidden-layer size in ANN, kernel type in SVM).  
The `caret::train()` function in R can automatically search combinations of these parameters to optimize a performance metric such as accuracy or RMSE.  
However, as the text warns, **better in-sample fit doesn’t guarantee better out-of-sample prediction**—computation cannot replace statistical judgment.

| Concept | Purpose | Caveat |
|:--|:--|:--|
| Grid or random search | Systematically test parameter combinations | Risk of over-tuning to the training data |
| `trainControl()` | Defines resampling method (CV, bootstrapping) | Small search spaces can miss global optimum |
| Model evaluation | Chooses “best” set by average metric | Must validate on unseen data |

**Relevance to Thesis:**  
Hyperparameter tuning can mask fragility: a model fine-tuned to one distribution might fail under a slightly shifted one. In my experiments, tuning will be separated from robustness testing to avoid biased results.

---

## Bootstrapping vs Cross-Validation  

Both methods resample data, but their goals differ:  

| Feature | Cross-Validation | Bootstrapping |
|:--|:--|:--|
| Sampling | Without replacement | With replacement |
| Purpose | Estimate out-of-sample error | Broaden data-process view / stabilize model |
| Typical use | Model evaluation | Model improvement (e.g., bagging) |

Bootstrapping mimics drawing many possible datasets from the same process. Each resample includes some observations multiple times and leaves out others, encouraging the model to generalize beyond a single sample.  

**Relevance to Thesis:**  
Bootstrapping can reveal how sensitive a model is to sample variability—an early indicator of robustness (or lack thereof) to domain shift.

---

## Ensemble Methods  

Ensembles combine multiple models to **leverage complementary strengths**.  
The idea: individual learners may fail on different parts of the data, but aggregated predictions tend to cancel errors.  

- **Homogeneous ensembles:** same model type with varied parameters  
- **Heterogeneous ensembles:** different algorithms (e.g., ANN + SVM + Tree)  

Competitions such as Netflix Prize and Kaggle confirm that ensembles consistently outperform single models.

**Relevance to Thesis:**  
By averaging diverse perspectives on the data, ensembles often yield **more robust predictions** under moderate distributional changes.

---

## Bagging (Bootstrap Aggregation)  

Bagging trains multiple versions of the same learner on bootstrapped samples and combines their predictions.  
It primarily reduces **variance** without greatly increasing bias.

### Random Forest (RF)  

A **Random Forest** is a bagged collection of CART trees with an additional randomization step:
1. Draw a bootstrapped sample of the data.  
2. At each split, consider only a random subset of predictors (≈ √p for classification, p/3 for regression).  
3. Repeat for many trees (e.g., 100+).  
4. Aggregate by **majority vote** (classification) or **average** (regression).

Key points:
- **Out-of-Bag (OOB) Error:** built-in estimate using observations excluded from each bootstrap sample.  
- **Variable Importance:** RF tracks how often variables appear in splits.  
- **Tuning Parameter:** number of predictors considered per node (mtry).  

| Strengths | Weaknesses |
|:--|:--|
| Handles high-dimensional, noisy data well | Can still overfit if too many trees or variables |
| Provides OOB estimate and variable importance | Harder to interpret than single trees |

**Relevance to Thesis:**  
RF’s ensemble structure inherently resists overfitting and noise, making it an excellent candidate for **robustness baselines** when testing shifted domains such as IMDB → Amazon Reviews.

---

## Boosting and XGBoost  

**Boosting** sequentially fits models, giving **higher weight** to previously misclassified or poorly predicted observations. Each new model corrects the errors of the prior one.

### XGBoost (Extreme Gradient Boosting)
- Introduces **gradient descent** to optimize an explicit loss (objective) function.  
- Uses efficient numerical methods (sparse matrices, regularization) for speed.  
- Supports internal train/test splitting and parallelization.  
- Parameter `colsample_bytree` borrows Random Forest’s idea of feature subsampling.

| Feature | Classic Boosting | XGBoost Enhancement |
|:--|:--|:--|
| Weighting | Ad-hoc up-weighting | Gradient-based optimization |
| Efficiency | Slow | Highly optimized, scalable |
| Regularization | Limited | Built-in L1/L2 penalties |
| Use cases | Modest accuracy gains | State-of-the-art in tabular tasks |

**Relevance to Thesis:**  
Boosting reduces bias but can increase variance if over-focused on training errors. Testing XGBoost on shifted data will highlight whether *bias correction* or *variance control* dominates robustness.

---

## Summary Comparison  

| Method | Core Idea | Strengths | Weaknesses | Robustness Implication |
|:--|:--|:--|:--|:--|
| **Automated Tuning** | Optimize hyperparameters via resampling | Improves fit | Can over-optimize to one sample | Must use shift-aware validation |
| **Bootstrapping** | Resample with replacement | Stabilizes estimates | Can inflate bias | Captures sample variability |
| **Bagging / RF** | Average models on bootstrapped data | Reduces variance; interpretable importance | Limited bias reduction | High variance-robustness |
| **Boosting / XGBoost** | Sequentially correct prior errors | Reduces bias; high accuracy | Sensitive to noise | Tests robustness under shift |

---

## Overall Connection to Thesis  

This chapter transitions from **evaluation** to **enhancement**—showing how strategic resampling and aggregation create models that **generalize beyond one dataset**.  
For my thesis, these techniques form the experimental backbone for testing **robustness under natural distribution shifts**:

- Compare single learners (ANN, SVM) vs. ensembles (RF, XGBoost).  
- Evaluate whether bagging’s variance reduction or boosting’s bias reduction yields higher OOD accuracy.  
- Examine how OOB error and variable-importance metrics align with robustness indicators.  

---

## Use of AI  

For this week’s notes, I summarized *Machine Learning with R (4th ed., Ch. 11)* and Professor Hoerl’s “Model Improvement” slides.  
I then used **OpenAI GPT-5** to help format, organize, and clarify the Markdown document.  
All interpretations and connections to robustness reflect my own understanding of the material.
