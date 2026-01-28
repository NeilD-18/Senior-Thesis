# 3 Methods

## 3.1 Study Design and Research Questions

This thesis evaluates the robustness of standard supervised learning models under two broad failure modes: degradation under controlled data corruption and degradation under distribution shift. The empirical study is organized around two “core” classification settings and one regression extension. The core experiments are (i) a tabular classification benchmark on the Adult Income dataset and (ii) a cross-domain text classification benchmark in which models are trained on IMDB movie reviews and evaluated on Amazon product reviews. The regression extension uses Airbnb price prediction to examine whether qualitative robustness trends persist when the target is continuous rather than categorical.

Across the core experiments, robustness is assessed by systematically varying (1) corruption type, (2) corruption severity, and (3) model class, while holding preprocessing, evaluation splits, and tuning protocol fixed. This produces degradation curves (performance vs. severity) for each (model, corruption) pair and enables direct comparisons of model families with respect to both nominal accuracy and sensitivity to data quality.

## 3.2 Datasets and Prediction Tasks

Adult Income (tabular classification). The Adult dataset predicts whether an individual’s income exceeds $50K using demographic and employment-related variables. The feature set contains both numeric and categorical predictors and includes missing values in the raw data. The learning task is binary classification.

IMDB and Amazon (text classification under domain shift). The IMDB dataset provides labeled movie reviews for in-domain training. The Amazon reviews serve as an out-of-domain evaluation set for the same binary sentiment task. This benchmark isolates a realistic distribution shift: vocabulary, writing style, and topical content differ substantially between movie reviews and product reviews, while the label semantics (positive vs. negative sentiment) remain consistent.

Airbnb (regression extension). The Airbnb dataset predicts nightly price from listing and host attributes. The feature space is mixed-type, similar in structure to Adult, but the target is continuous, allowing robustness analyses under corruptions in a regression context.

## 3.3 Preprocessing and Feature Representation

To ensure that robustness comparisons reflect model behavior rather than inconsistencies in data handling, preprocessing is standardized within each modality.

Tabular preprocessing (Adult, Airbnb). For tabular datasets, categorical predictors are encoded into a numeric representation suitable for classical ML models (e.g., one-hot encoding). Numeric predictors are scaled when required by the model class (notably for SVMs), using parameters fit on the training split only. Missing-value markers in the raw data are treated consistently as missing entries, and a single missing-data handling strategy is applied throughout the study so that differences across corruption settings are attributable to the corruption mechanisms rather than changing imputation policies.

Text preprocessing (IMDB, Amazon). Text inputs are converted to a fixed-dimensional vector representation. The representation is fit using the IMDB training data and then applied without modification to IMDB test data and the Amazon evaluation set. This is essential for the domain shift study: the same feature mapping is used across domains so that performance changes reflect distributional differences rather than changes in the representation itself.

## 3.4 Corruption and Shift Generation

Robustness is evaluated using both synthetic corruptions (where the mechanism and severity are controlled) and real domain shift (IMDB → Amazon). Corruptions are parameterized by a severity value $s$ that increases monotonically from the clean condition $s = 0$ to a maximally corrupted condition $s = s_{\max}$. Exact severity grids are fixed prior to final experiments and reported alongside results (or in an appendix) to support reproducibility.

### 3.4.1 Tabular corruptions (Adult; Airbnb where applicable)
Three corruption families are considered in the tabular setting.

Additive feature noise. Numeric predictors are perturbed by injecting zero-mean noise whose scale is controlled by severity. This simulates sensor error, measurement drift, and imperfect data collection pipelines.

Missingness. A fraction of entries are randomly removed (masked) according to severity, producing progressively less complete feature vectors. The downstream handling of missing values is kept fixed so that results measure sensitivity to missing information rather than sensitivity to changing preprocessing.

Class imbalance shift (Adult). To study robustness to changing class proportions, the training distribution is modified to create controlled imbalance levels while evaluation is performed on a fixed test distribution. This isolates the effect of imbalance in the training data and avoids conflating imbalance with changes in evaluation prevalence.

### 3.4.2 Text corruptions and domain shift (IMDB → Amazon)
Two mechanisms drive degradation in the text setting.

Token-level corruption. Synthetic text noise is introduced via token dropout (and, where used, substitution), parameterized by severity. Conceptually, this models incomplete text, typos, or loss of informative terms.

Cross-domain shift. Models trained on IMDB are evaluated on Amazon without target-domain fine-tuning. This captures a realistic domain shift where the input distribution changes in topic and lexical statistics while the task definition remains the same.

## 3.5 Model Classes and Training Protocol

The study compares three widely used model families that represent distinct inductive biases:

1. Random Forests (RF) as a bagged ensemble of decision trees,
2. Gradient Boosting (XGBoost-style boosting; XGB) as a boosted ensemble, and
3. Support Vector Machines (SVM) with both linear and RBF kernels.

Models are trained on the clean training data using a consistent tuning protocol (e.g., cross-validation or a held-out validation split). After selecting hyperparameters under the clean condition, hyperparameters are frozen and reused across all corruption severities. This is a key design choice: tuning separately at each severity can mask sensitivity by implicitly adapting the model to corruption, whereas freezing hyperparameters yields degradation curves that reflect the model’s inherent robustness.

## 3.6 Evaluation Setup and Metrics

For each dataset, performance is evaluated on held-out data not used in training or tuning. In the text domain shift setting, IMDB provides the in-domain test set (for sanity checking) while Amazon provides the out-of-domain test set (for the shift benchmark).

Classification metrics. Robustness is reported using Accuracy and F1-score, with AUROC included where scores are available and meaningful. F1-score is emphasized when class prevalence is altered, as it is less sensitive than accuracy to skewed label distributions.

Regression metrics. For Airbnb, performance is reported using RMSE and MAE to capture both squared-error sensitivity and absolute-error behavior.

## 3.7 Robustness Curves and Summary Robustness Measures

For each model class and corruption family, the primary result is a degradation curve that plots a performance metric $M(s)$ as a function of severity $s$. Degradation curves provide a more informative characterization than a single corrupted benchmark point: two models can have similar clean accuracy while differing substantially in how quickly performance collapses as corruption increases.

To support concise comparisons and ranking across models, each degradation curve is summarized using one or more scalar robustness measures. The primary summary used in this thesis is an area-under-curve style aggregate computed over the severity grid, which captures average performance across the corruption range rather than only worst-case behavior. Secondary summaries (reported when helpful) include worst-severity performance and slope-like degradation rates over the severity interval. Summary definitions are reported explicitly alongside results so that robustness claims are tied to a reproducible computation.

## 3.8 Uncertainty Quantification and Statistical Comparisons

Several sources of randomness can influence measured performance, including model stochasticity (notably for RF and boosting), corruption sampling, and (when applicable) subsampling used to induce class imbalance. To quantify uncertainty, each experimental condition is repeated over multiple random seeds, and results are aggregated by reporting mean performance and variability measures.

Where model comparisons are central (e.g., RF vs. XGB under a fixed corruption level), paired comparisons are used when feasible by sharing evaluation splits and corruption draws across models. Uncertainty in robustness summaries is estimated using resampling-based methods (e.g., bootstrap over runs and/or test instances), and differences are interpreted in light of these intervals rather than point estimates alone.

## 3.9 Reproducibility

All experiments are executed using a version-controlled codebase with configuration files that specify dataset versions, preprocessing choices, corruption parameters, model hyperparameters, and random seeds. Outputs include per-run logs and saved predictions sufficient to reproduce figures, tables, and robustness summaries. This ensures that robustness results can be regenerated end-to-end and that methodological choices are auditable.