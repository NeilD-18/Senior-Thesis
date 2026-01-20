# Week 4 – Classification Trees and Model Evaluation I

## Chapter 5 – Classification Trees

### Concept and Intuition
Classification and Regression Trees (CART) are **model-based** algorithms that use a series of **binary splits** on predictor variables (x’s) to classify observations into categories. The output variable (y) is typically **categorical**, and the algorithm recursively partitions the data to maximize separation between classes.

Each split is chosen to make the resulting groups as **homogeneous** as possible—meaning each leaf ideally contains observations from a single class. The process continues until the tree cannot be split further or a stopping rule (like minimum node size) is reached.

### How CART Works
At each node, the algorithm:
1. Searches across all variables and possible split points.  
2. Chooses the split that best separates the data.  
3. Repeats recursively for each child node.

This produces a tree-like structure where each internal node represents a decision, and each leaf node represents a class prediction. The first split is typically the most important, indicating the variable with the highest predictive power.

### Interpreting the Tree
One of the main advantages of CART is **interpretability**. You can easily trace how a decision was made by following the path from root to leaf.

- The **first split** indicates the most informative variable.  
- **Subsequent splits** capture secondary relationships and potential **interactions** between features.  
- If different branches use different splitting variables, it implies interactions (e.g., the effect of one variable depends on another).

### Splitting Criteria
CART uses measures like **entropy** or **Gini impurity** to decide the “best” split. Both quantify how mixed the classes are within a node.

Entropy = − Σ (pᵢ × log₂(pᵢ))

Where pᵢ is the proportion of class i in that node.  
- High entropy = mixed classes  
- Low entropy = pure node  

The goal is to choose the split that maximizes **information gain**:

Information Gain = Entropy(parent) − Weighted Average[Entropy(children)]

This ensures each split leads to the most informative partition possible.

### Overfitting and Pruning
Like many models, trees can **overfit** by growing too deep and memorizing the training data.  
To prevent this:
- Use **pruning**, which removes branches that don’t improve validation accuracy.  
- Use a **test set** or **cross-validation** to decide the optimal tree size.

A fully grown tree has perfect training accuracy but poor generalization to new data.

### Bagging, Boosting, and Random Forests
- **Bagging (Bootstrap Aggregation):** Builds multiple trees on resampled data and averages predictions to reduce variance.  
- **Boosting:** Sequentially reweights misclassified samples to focus on harder cases.  
- **Random Forests:** An ensemble of trees trained on random subsets of features, improving both stability and accuracy.

### Strengths and Weaknesses
| Strengths | Weaknesses |
|:--|:--|
| Highly interpretable and visual | Can overfit without pruning |
| Handles numeric and categorical data | Sensitive to small data changes |
| Captures nonlinear relationships | Can be unstable and high variance |

### Relevance to Thesis
Classification trees are particularly relevant for testing **robustness** because they are **highly sensitive to small changes in data**. Minor perturbations to feature values or missing data can change the splits entirely, resulting in a different model structure.  
In my thesis, analyzing how classification trees react to **noisy, incomplete, or shifted data** will demonstrate how model interpretability and reliability break down under non-ideal conditions. This helps illustrate the trade-off between simplicity and stability in model design.


---

## Chapter 10 – Model Evaluation I

### Concept and Intuition
A model that performs well on the training data may not perform well on new, unseen data. Therefore, model evaluation must consider both **accuracy** and **generalization**.  
Relying on accuracy alone can be misleading—especially when one class dominates (e.g., 95% “good loans”). A naïve model that always predicts the majority class can appear accurate without being useful.

To address this, we use additional metrics that better capture model quality under **class imbalance** and **error asymmetry** (e.g., false positives vs. false negatives).

### Confusion Matrix
The **confusion matrix** summarizes the model’s classification results:

|               | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

Each cell helps evaluate different aspects of performance. For example, false negatives are often more critical in medical or fraud detection applications.

### Core Metrics
From the confusion matrix, we calculate:

- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Sensitivity (Recall) = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- Precision = TP / (TP + FP)
- Negative Predictive Value (NPV) = TN / (TN + FN)

- **Sensitivity (Recall):** Ability to correctly identify positive cases.  
- **Specificity:** Ability to correctly identify negative cases.  
- **Precision:** Proportion of predicted positives that are actually positive.  
- **NPV:** Confidence that a negative prediction is correct.

Each metric tells a different story, depending on the problem context.

### F1 Score
When Precision and Recall are both important, we use the **F1 Score**, which is their harmonic mean:

F1 = 2 × (Precision × Recall) / (Precision + Recall)

If either Precision or Recall is low, F1 will drop sharply. It is particularly useful for imbalanced data where accuracy alone can be misleading.

### Kappa Statistic
Accuracy doesn’t account for how well a model performs relative to random chance. The **Kappa statistic** adjusts for this:

Kappa = (Pr(a) − Pr(e)) / (1 − Pr(e))

Where:
- Pr(a) = observed accuracy  
- Pr(e) = expected accuracy by random guessing  

Example:

Pr(a) = 0.84, Pr(e) = 0.825
Kappa = (0.84 − 0.825) / (1 − 0.825) = 0.086

Interpretation:
| Kappa Range | Agreement |
|:--|:--|
| 0–0.2 | Poor |
| 0.2–0.4 | Fair |
| 0.4–0.6 | Moderate |
| 0.6–0.8 | Good |
| 0.8–1.0 | Very Good |

Kappa provides a more balanced evaluation—showing how much better a model performs than random guessing.

### ROC Curve and AUC
The **ROC Curve (Receiver Operating Characteristic)** plots the **True Positive Rate (Recall)** against the **False Positive Rate (1 − Specificity)** for different cutoffs.  
The **AUC (Area Under the Curve)** quantifies how well the model distinguishes between classes.

- AUC = 0.5 → random guessing  
- AUC = 1.0 → perfect classification  

Rules of thumb:
| AUC Range | Model Quality |
|:--|:--|
| 0.5–0.6 | None |
| 0.6–0.7 | Poor |
| 0.7–0.8 | Fair |
| 0.8–0.9 | Good |
| 0.9–1.0 | Excellent |

AUC is independent of the cutoff threshold, making it ideal for comparing classifiers across different settings.

### Additional Metrics
- **Balanced Accuracy:** (Sensitivity + Specificity) / 2 – adjusts for class imbalance.  
- **Prevalence:** Proportion of positives in the dataset.  
- **Detection Rate:** TP / Total observations.  
- **Detection Prevalence:** (TP + FP) / Total observations.

There are many metrics available, but the key is to choose those that align with the problem’s priorities (e.g., false negatives matter more than false positives in healthcare).

### Relevance to Thesis
This chapter directly supports the **evaluation framework** for my robustness experiments. Metrics like **Kappa**, **F1**, and **AUC** provide deeper insight into how model quality changes under data corruption, imbalance, or distributional shifts.  
For instance:
- **Kappa** measures improvement over randomness when data become noisy.  
- **F1** balances the cost of false positives and false negatives in unbalanced or corrupted datasets.  
- **AUC** reflects how well a model maintains discriminative ability as the input distribution changes.

By applying these metrics to my experiments, I can more accurately quantify **robustness degradation** across models and datasets.

---

## Summary Comparison

| Chapter | Focus | Key Concepts | Relevance to Thesis |
|:--|:--|:--|:--|
| 5 | Classification Trees | Binary splits, entropy, information gain, overfitting, pruning | Demonstrates how small input changes can drastically alter tree structure and performance |
| 10 | Model Evaluation I | Confusion matrix, Kappa, F1, AUC, sensitivity/specificity | Provides robust metrics to quantify model performance under noise and imbalance |

---

### Overall Connection to Thesis
Both chapters contribute directly to understanding **robustness in machine learning**.  
Classification Trees show how model structure itself can be unstable under noisy data, while Model Evaluation provides the statistical tools to **measure that instability**.  
Together, they offer both the cause (model sensitivity) and the method (robust evaluation metrics) for analyzing how performance shifts under data corruption and distributional change.

---

## USE OF AI
For the purpose of these notes, I drafted initial summaries and bullet points in Google Docs based on the assigned readings and lecture slides. I then used OpenAI’s GPT-5 to help organize, rephrase, and format the content into a clean, consistent Markdown document. All substantive ideas and interpretations reflect my own understanding of the material.
