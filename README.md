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

---

## ✅ Current Project Status

- ✔ **Data preprocessing complete**
- ⏳ **Ready for Stage 3: Machine Learning Modeling & Evaluation**

---
