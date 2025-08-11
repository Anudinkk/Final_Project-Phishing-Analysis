# 🛡️ Phishing URL Detection Analysis

### A Machine Learning Approach to Phishing URL Detection  
**"Building a robust classification model to proactively identify and neutralize malicious phishing URLs using textual and domain-based features."**

---

## 📌 Project Overview

This project focuses on developing a machine learning classifier to accurately detect phishing URLs. Leveraging the **Phishing URL Dataset**, we preprocess a rich set of features—from URL length to domain characteristics—to create a model that can serve as a critical component of a web security system.  

The primary objective is to build a reliable, automated tool that identifies online threats in real-time.

---

## 🎯 Objectives

- Preprocess the real-world **PhiUSIIL** dataset.
- Drop irrelevant and high-cardinality string features.
- Handle high-cardinality categorical features such as **TLD** efficiently.
- Apply **one-hot encoding** and **feature scaling**.
- Split the data using a **stratified train/test split**.
- Prepare a clean dataset for machine learning modeling.

---

## 📁 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- **Name**: PhiUSIIL Phishing URL Dataset  
- **Task**: Binary Classification  
  - `0` = Phishing  
  - `1` = Legitimate

---

## 🔧 Stage 1: Data Collection & Exploration

- ✅ **Data Loading**: Imported directly using the `ucimlrepo` library.
- ✅ **Initial Inspection**: Reviewed structure, columns, data types, and target label distribution.
- ✅ **Column Review**: Evaluated all 52 features for relevance and datatype.

---

## 🧼 Stage 2: Preprocessing & Feature Engineering

### 🔹 Dropping Irrelevant Columns
The following columns were removed due to low predictive value or complexity:
- `FILENAME` – File identifier.
- `URL`, `Domain` – High-cardinality strings; information already captured in engineered features.
- `Title` – Unstructured text; excluded to maintain project scope.

### 🔹 Handling High-Cardinality Categorical Feature: `TLD`

- `TLD` had **695 unique values**.
- Strategy:
  - Calculate frequency for each TLD.
  - Group TLDs with frequency < 10 as `"Other"`.
- ✅ Reduced dimensionality while preserving signal.

### 🔹 Categorical Encoding

- Applied **one-hot encoding** to the modified `TLD` column.
- Used `drop_first=True` to avoid multicollinearity.

### 🔹 Feature Scaling

- Used `StandardScaler` to scale all numerical features.
- Scaler fitted **only on training data** to prevent data leakage.

### 🔹 Train/Test Split

- Performed an **80/20 split** with `stratify=y`.
- Ensured balanced distribution of phishing vs. legitimate URLs across both sets.



## 🚀 Stage 3: Model Building
The following classifiers were trained:
1. **Logistic Regression** – Linear baseline  
2. **Decision Tree Classifier** – Non-linear, interpretable model  
3. **Random Forest Classifier** – Ensemble of decision trees  
4. **Support Vector Machine (SVM)** – Optimal separating hyperplane  
5. **K-Nearest Neighbors (k-NN)** – Distance-based classification  
6. **Gradient Boosting Classifier** – Sequential boosting of weak learners  

---

## ⚙️ Stage 4: Hyperparameter Tuning
**Tool:** `GridSearchCV`  

### Tuned Models:
- Decision Tree
- Random Forest
- Gradient Boosting

### Example: Gradient Boosting Parameters Tuned:
- `n_estimators`
- `learning_rate`
- `max_depth`
- `subsample`

**Benefits of Tuning:**
- Maximized accuracy
- Reduced overfitting
- Automated best-parameter search

---

## 📊 Stage 5: Evaluation Metrics
- **Accuracy** – Overall correct predictions  
- **Classification Report** – Precision, recall, F1-score per class  
- **Confusion Matrix** – Breakdown of prediction outcomes  

---

## 📈 Stage 6: Model Performance Results
| Model Name                  | Accuracy   |
|-----------------------------|------------|
| **Random Forest Classifier**| **0.999958** |
| Decision Tree Classifier    | 0.999894   |
| Logistic Regression         | 0.999576   |
| K-Nearest Neighbors         | 0.997010   |
| Support Vector Machine (SVM)| 0.997010   |

---

## 🏆 Stage 7: Conclusion
Based on the evaluation, the **Random Forest Classifier** achieved the **highest accuracy (0.999958)**, indicating it is the most effective model for this dataset.  
Key reasons for its performance:
- Ability to handle both numerical and categorical features
- Robustness against overfitting due to ensemble averaging
- High interpretability via feature importance

**Final Decision:**  
The **Random Forest Classifier** is selected as the final model for phishing URL detection.  
It will be saved and integrated into a security pipeline for **real-time threat detection**.

---

## 🛠 Tech Stack
- **Language**: Python  
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn, scikit-learn  
- **Tools**: Jupyter Notebook
---

## ✅ Current Project Status

- ✔ **Model Prediction completed**

---
