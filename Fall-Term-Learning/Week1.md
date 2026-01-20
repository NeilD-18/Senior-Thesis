# Week 1 – Chapter Notes & Thesis Connections  
_Machine Learning (Lantz, 4th Edition) – Chapters 1 & 2_

---

## Chapter 1 – Introduction to Machine Learning

### What is Machine Learning?

- ML teaches computers to learn from data without explicit programming for every scenario.  
- Lantz frames ML as an attempt to **“automate the scientific method.”**  
  - Humans: **observation → hypothesis → experiment → refine**  
  - ML: **data → model → test on unseen data → update**  
- **Key limitation:** computers are still weak at *defining the right question* — problem framing remains human-driven.

---

### ML vs Traditional Statistics

| Statistics (Classical) | Machine Learning |
|------------------------|------------------|
| Explains underlying processes | Focuses on prediction accuracy |
| Relies on distributional assumptions (e.g., Normality) | Less concerned with assumptions |
| Emphasizes inference & significance testing | Uses complex black-box models (NNs, SVMs, trees) |
| Works with small/clean datasets | Thrives on large-scale data |

 -> **Big idea:** _Statistics = explanation. ML = prediction._

---

### The Black Box Issue

- Statisticians care about *how* variables influence outcomes.  
- ML practitioners often treat models as **black boxes** — if it predicts well, that’s “good enough.”  
- This creates philosophical tension between fields.

---

### Key Concepts

- **Generalization (ML) vs Inference (Stats):** different words for applying from sample to population.  
- **Overfitting:** model performs well on training data but poorly on unseen data.  
  - Root cause: excessive model complexity.  
  - Fix: **train/test split, cross-validation, regularization.**

---

### Types of ML Models

| Category | Description | Example |
|----------|-------------|---------|
| **Regression** | Predicts continuous outputs | GPA, stock price |
| **Classification** | Predicts classes | Spam / Not Spam |
| **Supervised Learning** | Training with labels | We have Y’s |
| **Unsupervised Learning** | No labels — discover structure | Clustering |

---

### Real-World Comparison (Slides Example – COVID-19)

| Statisticians | Data Scientists |
|---------------|-----------------|
| Small-scale controlled trials | Global-scale predictive models |
| Focus on inference | Focus on forecasting |

Both valid — just optimized for different goals.

---

### Chapter 1 Takeaway

> ML = prediction-first, black-box tolerant, data-hungry.  
> Statistics = inference-first, assumption-based, theory-driven.

**In my project:** Overfitting and generalization are *central* — I’m looking at how models **fail** when data shifts or degrades.

---

## Chapter 2 – Understanding Data (in R)

This chapter is foundational: how to **represent, clean, and manipulate data** — essential before any modeling.

---

### Data Types in R

| Type | Example | ML Relevance |
|------|---------|--------------|
| Numeric | 4.3, -7.1 | Continuous inputs |
| Integer | 5L | Count data |
| Character | "apple" | Text inputs |
| **Factor** | "Male", "Female" | Encodes categories |
| Logical | TRUE/FALSE | Binary flags |

-> **Factors are crucial** — they prevent treating categories as numbers.

---

### Data Structures in R

| Structure | Purpose |
|-----------|---------|
| Scalar | Single value |
| Vector | 1D sequence |
| **Data Frame** | Tabular (rows = observations, columns = variables) |
| List | Mixed-type container |

-> Almost all ML datasets come in **data frame** format.

Note: I'll be doing implementation in Python, but I figured I can include some stuff about R in here too

---

### Importing Data

- Manual entry (rare)
- **CSV files (most common)**
- Databases / JSON / Excel

**In my project:** I’ll be importing the **UCI Adult dataset** and **IMDB Reviews** — both CSV/text.

---

### Handling Missing Data

- Missing values = `NA`
- Key tools:
  - `is.na()` → detect  
  - `na.omit()` → drop rows  
  - `na.rm=TRUE` → ignore in calculations

Note: **Dropping rows can bias results** — this is important for the *robustness* experiments I’ll be running.

---

### Data Manipulation (dplyr)

| Command | Purpose |
|---------|---------|
| `select()` | pick columns |
| `filter()` | pick rows |
| `mutate()` | create new columns |
| `arrange()` | sort |
| `tidyr` | reshape |

These are my **surgical tools** for dataset preparation, corruption, and stress testing.

---

### Summarizing Data

- **Statistics:** mean, median, variance, SD  
- **Visuals:** boxplots, histograms, scatterplots  
- **Crosstabs:** categorical comparisons

---

### Chapter 2 Takeaway

> To test robustness, I need to control **how data is represented and degraded.**

- I’ll **inject missingness, noise, and distribution shifts** deliberately.  
- Mis-specified data types (e.g., numeric vs factor) could *silently* break models — part of what I need to watch for.

---

## Big Picture

| Chapter | Relevance to My Project |
|---------|--------------------------|
| **Chapter 1** | Shows why robustness matters → models overfit & fail when environments change. |
| **Chapter 2** | Gives me the tools → how to structure, manipulate, and corrupt datasets so I can test robustness systematically. |

---

## Final Reflection

Ch.1 gave me the theory: why generalization and robustness matter.  
Ch.2 gave me the mechanics: how to prepare and manipulate data.  
Together, they set the stage for my project — which is basically about *breaking these models on purpose* to see which ones hold up.
