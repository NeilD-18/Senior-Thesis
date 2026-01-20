# Winter Term Draft Schedule  
**Neil Daterao – Senior Thesis (Prof. Hoerl)**  
**Title:** *Robustness of Machine Learning Models to Distribution Shifts and Noisy Data*  

---

### Overview  

This term follows a **hybrid experimental design**:  
- A **factorial-style core experiment** on the **Adult Income** (tabular) and **IMDB → Amazon** (text, cross-domain) datasets to measure performance degradation across corruption types and severities.  
- An **individual exploratory component** using the **Airbnb Price Prediction** dataset to test robustness of regression models under noise and missingness.  

The plan emphasizes early model execution (Week 1) and steady progress toward analysis and thesis completion by Week 10.

---

| **Week** | **Focus / Objectives** | **Deliverables** |
|:--|:--|:--|
| **1 – Execution Kick-off** | Run baseline models on **Adult**, **IMDB**, and **Airbnb** datasets. Verify preprocessing, logging, and reproducibility. Meet with Prof. Hoerl to confirm milestones. | Clean baselines (RF, XGB, SVM); initial accuracy/F1/RMSE tables. |
| **2 – Methodology Write-up & Testing** | Draft **Methods** section describing datasets, models, and metrics. Conduct initial dry-run experiments to confirm runtime feasibility. | Drafted Methods section; early baseline plots. |
| **3 – Finalize Experimental Setup** | Complete corruption modules for tabular, text, and regression data. Confirm seed control, evaluation metrics, and configuration files. | Validated corruption scripts; finalized config schema. |
| **4 – Pilot and Validation** | Run small pilot grid (limited models × severities) on **Adult** and **IMDB**. Run first Airbnb noise/missingness pilot. | Pilot results summary; finalized factor levels. |
| **5 – Stability & Bias–Variance Checks** | Begin ensemble comparison (RF vs XGBoost) across all datasets. Evaluate run-to-run stability and bias–variance behavior. | Mid-term progress memo; stability plots. |
| **6 – Tabular Domain Experiments** | Execute **Adult** dataset robustness tests (Gaussian noise, missingness, imbalance). Compute Accuracy, F1, AUROC. | Aggregated tabular results; draft degradation curves. |
| **7 – Text Domain Experiments** | Run **IMDB → Amazon** token-dropout experiments. Compare RF, XGB, and SVM (RBF vs linear). | Text robustness results; model comparison summary. |
| **8 – Regression Robustness & Full Analysis** | Extend robustness tests to **Airbnb** (noise + missingness). Aggregate all results and generate final figures (degradation curves, robustness AUC, calibration plots). Begin **Results** section. | Figures and tables ready; Results section draft. |
| **9 – Discussion & Final Edits** | Write **Discussion** and **Conclusion** sections. Interpret findings across all three datasets, finalize visuals, and integrate advisor feedback. | Complete Discussion/Conclusion drafts; slide outline. |
| **10 – Final Submission** | Submit full thesis. Ensure reproducibility of all code and results. | Final thesis, polished figures, code archive.|

---

### Notes  
- **Adult** and **IMDB → Amazon** = core factorial experiment datasets.  
- **Airbnb** = targeted regression deep-dive within the hybrid framework.  
- Writing begins early (Week 2) and continues iteratively.  
- Continuous experimentation from Week 1 ensures a complete, analyzable dataset well before Week 10.
