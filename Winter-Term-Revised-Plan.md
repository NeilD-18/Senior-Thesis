# **Winter Term Thesis Plan**

**Neil Daterao**

Senior Thesis – Mathematics & Statistics

Advisor: Prof. Roger Hoerl

---

## **Title**

**Robustness of Machine Learning Models to Distribution Shifts and Noisy Data**

---

## **1. Project Overview**

This senior thesis investigates the robustness of common machine learning models under distribution shifts and data corruption. The goal is to empirically evaluate how model performance degrades as a function of noise, missingness, imbalance, and cross-domain transfer, and to compare robustness behavior across model classes and data modalities.

The project follows a **hybrid experimental design**:

- A **core factorial-style experiment** using:
    - The **Adult Income** dataset (tabular classification)
    - The **IMDB → Amazon Reviews** datasets (text classification under cross-domain shift)
- A **targeted exploratory regression study** using:
    - The **Airbnb Price Prediction** dataset

The Adult and IMDB → Amazon experiments form the primary basis for structured robustness comparisons. The Airbnb study is intentionally treated as a qualitative extension to regression tasks, focusing on noise and missingness rather than a full factorial design.

---

## **2. Research Objectives**

The thesis is guided by the following core research questions:

1. How does predictive performance degrade as a function of corruption type and severity?
2. Do ensemble-based models (e.g., Random Forests, Gradient Boosting) exhibit greater robustness than margin-based or kernel-based models?
3. Are robustness patterns under synthetic corruption qualitatively similar to those observed under real cross-domain shifts?
4. To what extent do robustness trends observed in classification settings extend to regression problems?

---

## **3. Datasets and Modeling Framework**

### **3.1 Datasets**

- **Adult Income**: Tabular classification dataset used to study robustness under Gaussian noise, missingness, and class imbalance.
- **IMDB → Amazon Reviews**: Text classification datasets used to evaluate cross-domain robustness via token-level corruption and domain shift.
- **Airbnb Price Prediction**: Regression dataset used for exploratory analysis of robustness to noise and missing data.

### **3.2 Models**

Across datasets, the following model families will be evaluated:

- Random Forests (RF)
- Gradient Boosting / XGBoost (XGB)
- Support Vector Machines (SVM), including linear and RBF kernels

### **3.3 Evaluation Metrics**

- **Classification**: Accuracy, F1-score, AUROC
- **Regression**: RMSE and MAE

In addition to point metrics, degradation curves and summary robustness measures will be computed across corruption severities.

---

## **4. Experimental Design**

### **4.1 Core Factorial Experiments**

For Adult Income and IMDB → Amazon, experiments will systematically vary:

- Corruption type (noise, missingness, imbalance, token dropout)
- Corruption severity
- Model class

Controlled random seeds and standardized configurations will ensure reproducibility and enable stability analysis.

### **4.2 Exploratory Regression Study**

The Airbnb dataset will be used to assess whether robustness patterns observed in classification settings extend to regression. This component emphasizes interpretability and qualitative comparison rather than exhaustive factorial coverage.

---

## **5. Week-by-Week Timeline**

| **Week** | **Focus** | **Tasks** | **Deliverables** |
| --- | --- | --- | --- |
| 1 | Execution Kick-off | Run baseline models on Adult, IMDB, and Airbnb datasets; verify preprocessing pipelines, logging, and reproducibility; meet with advisor to confirm milestones | Baseline accuracy/F1/RMSE tables |
| 2 | Methodology Drafting and Dry Runs | Draft Methods section describing datasets, models, metrics, and evaluation protocols; conduct small-scale dry runs to confirm computational feasibility | Draft Methods section; preliminary baseline plots |
| 3 | Experimental Design Finalization | Complete corruption modules for tabular, text, and regression data; finalize configuration schemas, random seed control, and evaluation procedures; lock research questions and experimental scope | Validated corruption scripts; finalized experimental configuration |
| 4 | Pilot Experiments and Validation | Run pilot grids on Adult and IMDB datasets; conduct initial Airbnb noise and missingness experiments; validate factor levels and runtime expectations | Pilot results summary; confirmed experimental design |
| 5 | Stability and Bias–Variance Analysis | Compare ensemble models across datasets; evaluate run-to-run stability and variance under corruption | Mid-term progress memo; stability and variance plots |
| 6 | Tabular Domain Experiments | Execute full robustness experiments on Adult Income dataset; analyze degradation under noise, missingness, and imbalance | Aggregated tabular results; draft degradation curves |
| 7 | Text Domain Experiments | Run IMDB → Amazon robustness experiments; compare SVM (linear vs RBF), Random Forests, and XGBoost | Text robustness results; model comparison summary |
| 8 | Regression Robustness and Full Analysis | Complete Airbnb robustness experiments; aggregate results across all datasets; generate final figures and tables; begin drafting Results section | Final figures and tables; Results draft |
| 9 | Discussion and Final Revisions | Write Discussion and Conclusion sections; interpret findings across datasets and model classes; address limitations and threats to validity | Complete Discussion and Conclusion drafts; presentation outline |
| 10 | Final Submission | Finalize thesis document; verify full reproducibility of code and results | Final thesis; code archive |

---

## **6. Scope and Risk Management**

- Writing begins early and proceeds iteratively alongside experimentation
- Pilot experiments serve as a formal design validation checkpoint
- The Airbnb study is scoped to avoid overgeneralization
- All experiments prioritize reproducibility and interpretability

---

## **7. Expected Outcome**

The thesis will produce a systematic empirical comparison of robustness behaviors across datasets, model classes, and data modalities, providing both quantitative degradation analyses and qualitative insights into model reliability under imperfect data conditions.

# USE OF AI
In this document, the content was all generated by me, however proof reading and structure was enhanced by Notion's built in AI-tool
