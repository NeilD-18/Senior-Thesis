# Senior Thesis Proposal

- **Name:** Neil Daterao
- **Advisor:** Professor Hoerl
- **Tentative Title:** *Robustness of Machine Learning Models to Distribution Shifts and Noisy Data*

---

## Motivation and Background

Machine learning models are increasingly deployed in real-world applications such as finance, healthcare, and recommendation systems. A persistent challenge, however, is their **lack of robustness**: models often perform well on training and validation data but degrade when faced with **distribution shifts** or **data imperfections** such as missing values, mislabeled data, or noisy features.

For example, a census-income classifier trained on one demographic distribution may fail when tested on a population with different class proportions. Likewise, a sentiment classifier trained on IMDB movie reviews often performs poorly when applied to product reviews from Amazon, despite both being binary text classification tasks.

This project will systematically evaluate how standard machine learning algorithms degrade under such conditions and compare techniques to improve robustness. By focusing on two canonical classification tasks—**income classification (tabular data)** and **sentiment analysis (text)**—the project will highlight vulnerabilities of widely used models and explore methods to mitigate them.

---

## Problem Definition

The central research question is:

> *Given a classification task, how does the performance of different machine learning algorithms degrade under data corruption and distribution shifts, and which mitigation strategies are most effective at preserving accuracy?*

* **Task framing:** Binary classification.
* **Datasets:**

  * **[UCI Adult Income dataset](https://archive.ics.uci.edu/dataset/2/adult)**  — classify whether an individual’s income exceeds \$50k/year.
  * **[IMDB Movie Reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/)** — train sentiment models on movie reviews.
  * **[Amazon Reviews](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/) (Multi-Domain Sentiment dataset)** — test models cross-domain to evaluate robustness.
  * **[Airbnb Price Prediction](https://www.kaggle.com/competitions/airbnb-price-prediction/data)** - predict the price of Airbnb rentals
* **Algorithms compared:** KNN, Naïve Bayes, Decision Trees (CART), Random Forests, Gradient Boosted Trees (XGBoost), Support Vector Machines (SVM), shallow Neural Networks.
* **Robustness challenges studied:**

  * **Noise:** Gaussian perturbations on numeric features (Adult), token dropout/substitution in text.
  * **Missingness:** Randomly dropping values (tabular) or words (text).
  * **Distribution shift:** Class imbalance in Adult; cross-domain transfer (IMDB → Amazon).

---

## Plan Across Two Terms

### **Fall Term (Background and Foundations)**

The fall will be **front-loaded** with theory and practice using *Machine Learning with R* (Lantz, 4th ed.) and STA-364 slides:

1. **Literature Review:** Read and summarize 4–5 key papers on robustness and domain shift.

   * Hendrycks & Dietterich (2019): corruption benchmarks.
   * Rolnick et al. (2017): label noise robustness.
   * Arjovsky et al. (2020): out-of-distribution generalization.
   * Szegedy et al. (2014): adversarial examples.
   * Wang et al. (2021): survey on OOD robustness.

2. **Dataset preparation:** Preprocess Adult and IMDB datasets; acquire Amazon reviews for testing.

3. **Baseline models:** Train initial classifiers (LogReg, Decision Trees, Random Forests, SVMs, Naïve Bayes) on **clean data** to establish baseline accuracy and F1.

4. **Experiment design:** Finalize corruption protocols (noise levels, missingness %, label noise rates, token dropout, domain shifts).

---

### **Winter Term (Experiments and Analysis)**

1. **Robustness benchmarking:**

   * Adult dataset: introduce feature noise, missingness, and label flips; evaluate model degradation.
   * IMDB → Amazon: evaluate cross-domain performance and token-level corruptions.

2. **Mitigation strategies:**

   * Regularization (L1/L2, dropout).
   * Data augmentation (synonym replacement, token dropout for text).
   * Class reweighting for imbalance.
   * Compare effectiveness of strategies.

3. **Evaluation:**

   * Report accuracy, F1 score, AUROC.
   * Plot robustness curves (performance vs. corruption severity).
   * Use bootstrap resampling and statistical tests for comparisons.

4. **Deliverables:**

   * Final thesis document with methods, results, and analysis.
   * Public GitHub repository with reproducible code.
   * Visualizations of robustness tradeoffs.

---

## Expected Outcomes

* A **benchmark study** of how standard ML algorithms degrade under data imperfections and distribution shifts.
* Insights into **which models are inherently more robust** across tabular and text domains.
* Evaluation of **simple mitigation strategies** and their effectiveness.
* A reproducible pipeline and thesis write-up that can serve as a foundation for future work in robustness and reliability research.

---

## Why This Project

This project is academically rigorous while being tightly scoped to two canonical datasets. It aligns with the development plan and provides hands-on experimentation with both classical ML and lightweight neural methods. Beyond the classroom, the project builds a **portfolio-ready contribution** to robustness research, directly relevant to ML/AI roles at research-driven organizations like OpenAI and Anthropic, where distribution shift and reliability are critical concerns.

---

## References (Initial Reading List)

1. Hendrycks, D., & Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. [Paper link](https://arxiv.org/abs/1903.12261)
2. Rolnick, D., Veit, A., Belongie, S., & Shavit, N. (2017). *Deep Learning is Robust to Massive Label Noise*. [Paper link](https://arxiv.org/abs/1705.10694)
3. Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2020). *Invariant Risk Minimization*. [Paper link](https://arxiv.org/abs/1907.02893)
4. Szegedy, C., Zaremba, W., Sutskever, I., et al. (2014). *Intriguing Properties of Neural Networks*. [Paper link](https://arxiv.org/abs/1312.6199)
5. Wang, H., Ye, F., Li, Z., & Song, M. (2021). *Generalizing to Unseen Domains: A Survey on Domain Generalization*. [Paper link](https://arxiv.org/abs/2103.03097)

