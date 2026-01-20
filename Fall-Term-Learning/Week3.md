# Week 3 – Nearest Neighbors (k-NN) and Naïve Bayes

## Chapter 3 – k-Nearest Neighbors (k-NN)

### Concept and Intuition
k-Nearest Neighbors (k-NN) is a **classification** algorithm that assigns a label to a new observation based on its **proximity to existing labeled examples**. It assumes discrete output (class) variables and continuous input features. No explicit model is built—this is often called **“lazy learning”** because computation occurs at prediction time.

The algorithm measures how “close” a test instance is to all training instances using a distance metric (typically **Euclidean distance**), then assigns the most common class among the **k** nearest examples.

### Distance and Similarity
The most common metric is Euclidean distance:

`dist(p,q) = sqrt((p1 - q1)^2 + (p2 - q2)^2 + ... + (pn - qn)^2)`

Other metrics include Manhattan (L1) and Minkowski distances.  
The choice of distance metric—and ensuring all features are on a comparable scale—is essential for accurate classification.

### Choosing k
The value of k controls model bias and variance:
- **Small k:** low bias, high variance (overfitting)
- **Large k:** high bias, low variance (oversmoothing)
  
A rule of thumb is k = √n, where n is the number of samples.  
Model stability can be tested by evaluating performance across multiple k values.

### Preprocessing and Feature Scaling
Since distance is sensitive to feature magnitudes, standardizing inputs is crucial:
- **Normalization:** rescales all features to the range [0, 1]
- **Z-score standardization:** rescales features to have mean 0 and standard deviation 1  
- **Dummy variables:** used to convert categorical features into numeric format

Without scaling, features measured in large units (like income in dollars) can dominate distance computations.

### Model Evaluation
Data are typically split into **training** and **test** sets (e.g., 80/20).  
Accuracy is evaluated using a **confusion matrix**, tracking True Positives, False Positives, True Negatives, and False Negatives.

If test accuracy is substantially lower than training accuracy, this indicates **overfitting**.  
To improve generalization, adjust k or use cross-validation for a more reliable performance estimate.

### Strengths and Weaknesses
| Strengths | Weaknesses |
|:--|:--|
| Simple and intuitive | Computationally heavy at prediction time |
| Works with nonlinear class boundaries | Sensitive to feature scaling |
| No assumptions about data distribution | Performs poorly with noise or irrelevant features |

### Relevance to Thesis
k-NN provides a natural framework for studying **robustness to noise, scaling, and missing data**. Because it directly relies on the geometry of the data, it’s highly sensitive to outliers and feature distortion—making it ideal for testing how model accuracy changes under perturbations or distributional shifts.  
In the context of the thesis, analyzing k-NN performance under feature corruption or incomplete data can illustrate how local similarity methods deteriorate with increasing noise.


---

## Chapter 4 – Naïve Bayes

### Core Idea
Naïve Bayes is a **probabilistic classification algorithm** based on **Bayes’ Theorem**:
P(C | X) = [ P(X | C) × P(C) ] / P(X)

It calculates the probability that a sample belongs to a class \(C\) given a set of features \(X\), then classifies the observation according to the highest posterior probability.

The method is called “naïve” because it **assumes independence among features**—an assumption rarely true but surprisingly effective in practice.

### Example – Spam Detection
Consider the task of identifying spam emails using words such as “Viagra,” “money,” and “unsubscribe.”  
Each word acts as a binary feature (present/absent), and we compute:

P(spam | W₁, W₂, ..., Wₙ) ∝ P(spam) × Π P(Wᵢ | spam)

The denominator \(P(X)\) is the same for all classes, so we compare only the numerators.  
If P(spam | B)  > P(ham | B), classify the email as spam.

### Dealing with Zero Probabilities
When a word never appears in spam (or ham), its conditional probability becomes zero, nullifying the whole product.  
To fix this, we use **Laplace smoothing** (add 1 to every count).  
This prevents zero probabilities and improves generalization on unseen data.

### Handling Continuous Variables
Naïve Bayes inherently works with categorical features.  
For continuous variables (like “length of email”), we can:
- **Discretize** the variable into bins, or  
- Assume a **normal distribution** and use probability densities (as implemented in R’s `naivebayes` function).  
Discretization is preferred when normality cannot be justified.

### Probability Normalization
Because the denominator \(P(X)\) is omitted during proportional calculations, computed scores are not true probabilities until normalized:

P(C | X) = [ P(X | C) × P(C) ] / Σ [ P(X | C′) × P(C′) ]

This step ensures all class probabilities sum to 1.

### Strengths and Weaknesses
| Strengths | Weaknesses |
|:--|:--|
| Fast, simple, interpretable | Unrealistic independence assumption |
| Performs well with small or noisy data | Struggles with continuous features unless discretized |
| Effective with high-dimensional text | Probability outputs not always well-calibrated |

### Relevance to Thesis
Naïve Bayes allows controlled testing of **robustness under assumption violations and data noise**.  
Because it explicitly models probabilities, we can analyze how accuracy and posterior confidence degrade when correlations among features are introduced or when class priors shift.  
This supports the thesis goal of quantifying model stability under distributional changes.  
Additionally, comparing smoothed vs. unsmoothed (Laplace = 1 vs 0) versions demonstrates how simple regularization improves generalization to unseen, noisy data.

---

## 🧩 Summary Comparison

| Aspect | k-NN | Naïve Bayes |
|:--|:--|:--|
| Type | Instance-based / lazy | Probabilistic / model-based |
| Training cost | Low | Moderate |
| Prediction cost | High (distance computations) | Low (closed-form calculation) |
| Assumptions | None about data distribution | Conditional independence |
| Typical use case | Numeric, geometric data | Text or categorical data |
| Robustness relevance | Noise, scaling, missing values | Independence violations, smoothing, class imbalance |

---

### 🧠 Overall Connection to Thesis
Both algorithms illustrate **contrasting approaches to classification**—one geometric, one probabilistic—providing complementary perspectives for robustness testing.  
By systematically adding noise, altering feature correlations, or shifting priors, we can quantify how each method’s underlying assumptions impact stability and generalization.  
Together, they form the foundation for understanding how data characteristics influence model reliability under distributional stress.
