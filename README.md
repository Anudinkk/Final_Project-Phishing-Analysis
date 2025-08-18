# 🛡️ Phishing URL Detection using Machine Learning  

Building a robust classification model to proactively identify and neutralize malicious phishing URLs using textual and domain-based features.  

---

## 📌 Project Overview  
This project focuses on developing a **machine learning classifier** to accurately detect phishing URLs.  
Leveraging the **PhiUSIIL Phishing URL Dataset**, we preprocess a rich set of features—from URL length to domain characteristics—to create a model that can serve as a critical component of a web security system.  

**Objective:** Build a reliable, automated tool that identifies online threats in real-time.  

---

## 🎯 Objectives  
- Preprocess the real-world PhiUSIIL dataset  
- Drop irrelevant and high-cardinality string features  
- Handle high-cardinality categorical features such as TLD efficiently  
- Apply one-hot encoding and feature scaling  
- Split the data using a stratified train/test split  
- Prepare a clean dataset for machine learning modeling  

---

## 📁 Dataset  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/)  
- **Name:** PhiUSIIL Phishing URL Dataset  
- **Task:** Binary Classification  
  - `0 = Phishing`  
  - `1 = Legitimate`  

---

## 🔧 Stage 1: Data Collection & Exploration  
- ✅ Loaded data using `ucimlrepo`  
- ✅ Inspected structure, columns, data types, and label distribution  
- ✅ Evaluated all 52 features for relevance  

---

## 🧼 Stage 2: Preprocessing & Feature Engineering  

### Dropping Irrelevant Columns  
- `FILENAME` – File identifier  
- `URL`, `Domain` – High-cardinality strings (info already captured in engineered features)  
- `Title` – Unstructured text  

### Handling High-Cardinality Feature: TLD  
- 695 unique values  
- Grouped TLDs with frequency `< 10` as `"Other"`  
- Result: Reduced dimensionality while preserving useful signal  

### Encoding and Scaling  
- One-hot encoding applied to categorical feature (TLD)  
- StandardScaler applied to numerical features  
- Train/Test split: 80/20 with stratify=y  

---

## 🚀 Stage 3: Model Building  
Trained multiple classifiers:  
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (k-NN)  
- Gradient Boosting Classifier  

---

## ⚙️ Stage 4: Hyperparameter Tuning  
- Tool: `RandomizedSearchCV`  
- Best Model: **Random Forest**  
- Best Accuracy: **0.9999**  

---

## 📊 Stage 5: Evaluation Metrics  
- Accuracy  
- Classification Report  
- Confusion Matrix  

---

## 📈 Stage 6: Model Performance Results  

| Model                        | Accuracy   |
|------------------------------|------------|
| 🌳 Random Forest Classifier  | **0.999915** |
| 🌲 Decision Tree Classifier  | 0.999894   |
| ➗ Logistic Regression       | 0.999576   |
| 📍 K-Nearest Neighbors       | 0.997010   |
| ⚖️ Support Vector Machine    | 0.997010   |

### Final Model: Random Forest  

**Accuracy:** 0.999915  

**Classification Report:**  

          precision    recall  f1-score   support
       0       1.00      1.00      1.00     23464
       1       1.00      1.00      1.00     23464

accuracy                           1.00     46928



**Confusion Matrix:**  

|               | Predicted Phishing | Predicted Legitimate |
|---------------|--------------------|----------------------|
| Actual Phishing   | ✅ 23464 | ❌ 0   |
| Actual Legitimate | ❌ 4     | ✅ 23460 |

---

## 🏆 Stage 7: Conclusion  
- The **Random Forest Classifier** achieved the highest accuracy.  
- Key strengths:  
  - Handles numerical + categorical features effectively  
  - Robust against overfitting  
  - Feature importance for interpretability  

✅ Final model saved as `random_forest_model.pkl` for deployment in a real-time threat detection pipeline.  

---

## 🛠️ Tech Stack  
- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn  
- **Tools:** Jupyter Notebook  

---

 
