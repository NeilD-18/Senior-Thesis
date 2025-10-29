# Week 5 – Black Box Methods: Neural Networks and Support Vector Machines

## Concept and Intuition

**Black box methods** are named "Black Box" because their internal workings are complex and difficult to interpret. Unlike simpler models such as linear regression or decision trees, these models rely on **nonlinear, multi-layered calculations**, which make them extremely **flexible and powerful**, but also **prone to overfitting** and **hard to explain**.  

In practical applications, we must balance **accuracy** with **interpretability**—especially when the goal is to understand relationships in the data, not just make predictions.  

---

## Artificial Neural Networks (ANNs)

### Structure and Function

An **Artificial Neural Network (ANN)** models relationships between predictors and outcomes by mimicking the **structure of the human brain**. It uses interconnected units called **neurons** (or **nodes**) organized into **layers**:
1. **Input layer** – represents features (x’s)
2. **Hidden layers** – transform weighted inputs through nonlinear functions
3. **Output layer** – produces the final prediction

Each neuron computes a **linear combination** of inputs and passes it through an **activation function** to determine its output signal. Common activation functions include:
- **Sigmoid** – smooth S-shaped curve between 0 and 1  
- **Tanh** – outputs between -1 and +1  
- **ReLU** (Rectified Linear Unit) – outputs 0 for negatives and the input value for positives  

The choice of activation function affects how the network learns and models nonlinear relationships.  

### Network Topology
The **topology** defines how many layers and neurons are used:
- **Single-layer (no hidden layer):** Equivalent to linear or logistic regression  
- **One or more hidden layers:** Allows modeling of nonlinear relationships  
- **Deep Neural Networks (DNNs):** Multiple hidden layers; high flexibility but high risk of overfitting  

Empirical research suggests that **2–3 hidden layers** are usually sufficient; adding more often leads to **divergence or overfitting**.  

### Training: Backpropagation
Neural networks learn using **backpropagation**, an iterative optimization process that adjusts weights to minimize the **sum of squared errors**. Training involves:
1. Forward pass – computing predictions  
2. Backward pass – updating weights based on errors  
3. Repeating over multiple **epochs** until convergence  

Training is **computationally intensive**, especially for large networks, and convergence is not guaranteed.  

### Strengths and Weaknesses

| Strengths | Weaknesses |
|:--|:--|
| Can approximate almost any nonlinear function | Extremely slow and computationally demanding |
| Highly flexible, few assumptions about data | Very prone to overfitting and poor generalization |
| Can handle complex relationships | Hard to interpret (“black box”) |

### Relevance to Thesis
ANNs are ideal for robustness testing because of their **sensitivity to noise** and **lack of interpretability**. Small perturbations in data (e.g., missing or mislabeled points) can drastically change weights and predictions, illustrating the instability of complex nonlinear models under distributional shifts.  

---

## Support Vector Machines (SVMs)

### Concept
**Support Vector Machines** (SVMs) are supervised learning methods that separate classes using a **hyperplane** that maximizes the **margin** between groups.  
Unlike decision trees that make axis-parallel splits, SVMs find the **widest possible boundary** in any direction.  

Key components:
- **Support vectors** – data points closest to the separating hyperplane; define the boundary  
- **Margin** – distance between the hyperplane and the nearest support vectors  

This approach ensures robust classification that generalizes well, even in high-dimensional spaces.  

### Handling Nonlinear Relationships
When classes are not linearly separable, SVMs use **kernel functions** to project data into a higher-dimensional space where a linear split becomes possible—a technique called the **kernel trick**.  

Common kernels:
- Linear: K(xᵢ, xⱼ) = xᵢ · xⱼ
- Polynomial: K(xᵢ, xⱼ) = (xᵢ · xⱼ + 1)ᵈ
- Radial Basis Function (RBF): K(xᵢ, xⱼ) = exp(−|xᵢ − xⱼ|² / c)
- Sigmoid: K(xᵢ, xⱼ) = tanh(a(xᵢ · xⱼ) − b)

Each kernel allows SVMs to model different kinds of nonlinear patterns.  

### SVM for Prediction
SVMs can also be used for regression by finding the **best-fitting hyperplane** in continuous response problems. This version is similar to linear regression but includes penalties for points outside a specified error margin.  

### Strengths and Weaknesses

| Strengths | Weaknesses |
|:--|:--|
| Works for both classification and prediction | Slow to train on large datasets |
| Not overly sensitive to noise or outliers | Hard to interpret (“black box”) |
| Less prone to overfitting than ANNs | Requires tuning of kernel and cost parameters |  

### Relevance to Thesis
SVMs are crucial for robustness analysis because their **kernel-based flexibility** allows them to model complex decision boundaries that adapt (or fail to adapt) under data shifts.  
By comparing SVM and ANN performance on noisy or shifted data, I can assess which method **maintains predictive stability** and which overfits to spurious patterns.

---

## Summary Comparison

| Method | Key Idea | Strengths | Weaknesses | Relevance to Thesis |
|:--|:--|:--|:--|:--|
| **ANN** | Layers of nonlinear transformations of inputs | Highly flexible, universal approximator | Prone to overfitting, hard to interpret | Tests robustness to noise and instability of complex models |
| **SVM** | Finds hyperplane maximizing margin, uses kernels for nonlinear splits | Accurate and less overfit-prone | Requires careful tuning, hard to interpret | Tests robustness of margin-based classifiers to data shifts |

---

### Overall Connection to Thesis
Both ANN and SVM represent **powerful but opaque learning systems**.  
By introducing controlled data corruption and distributional changes, I can evaluate:
- How much accuracy each loses under perturbations.  
- How model structure (layers, kernels) affects **robustness**.  
- Whether simpler interpretable alternatives might be preferable for stable performance under real-world uncertainty.  

---

## USE OF AI
For this week’s notes, I drafted outlines from *Machine Learning with R* and Professor Hoerl’s slides. I then used OpenAI’s GPT-5 to refine, structure, and format the material in Markdown. All interpretations and insights reflect my understanding of the readings and lecture content.
