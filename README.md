🛡️ Phishing URL Detection Analysis
A Machine Learning Approach to Phishing URL Detection

Building a robust classification model to proactively identify and neutralize malicious phishing URLs using textual and domain-based features.

📌 Project Overview
This project focuses on developing a machine learning classifier to accurately detect phishing URLs. Leveraging the Phishing URL Dataset, we preprocess a rich set of features—from URL length to domain characteristics—to create a model that can serve as a critical component of a web security system.

The primary objective is to build a reliable, automated tool that identifies online threats in real-time.

🎯 Objectives
 Preprocess the real-world PhiUSIIL dataset

 Drop irrelevant and high-cardinality string features

 Handle high-cardinality categorical features such as TLD efficiently

 Apply one-hot encoding and feature scaling

 Split the data using a stratified train/test split

 Prepare a clean dataset for machine learning modeling

📁 Dataset
Source: UCI Machine Learning Repository

Name: PhiUSIIL Phishing URL Dataset

Task: Binary Classification

0 = Phishing

1 = Legitimate

🔧 Stage 1: Data Collection & Exploration
✅ Data Loading: Imported directly using the ucimlrepo library

✅ Initial Inspection: Reviewed structure, columns, data types, and target label distribution

✅ Column Review: Evaluated all 52 features for relevance and datatype

🧼 Stage 2: Preprocessing & Feature Engineering
🔹 Dropping Irrelevant Columns
The following columns were removed due to low predictive value or complexity:

FILENAME – File identifier

URL, Domain – High-cardinality strings; information already captured in engineered features

Title – Unstructured text; excluded to maintain project scope

🔹 Handling High-Cardinality Categorical Feature: TLD
Challenge: TLD had 695 unique values

Strategy:

Calculate frequency for each TLD

Group TLDs with frequency < 10 as "Other"

✅ Result: Reduced dimensionality while preserving signal

🔹 Categorical Encoding
Applied one-hot encoding to the modified TLD column

Used drop_first=True to avoid multicollinearity

🔹 Feature Scaling
Used StandardScaler to scale all numerical features

Scaler fitted only on training data to prevent data leakage

🔹 Train/Test Split
Performed an 80/20 split with stratify=y

Ensured balanced distribution of phishing vs. legitimate URLs across both sets

🚀 Stage 3: Model Building
The following classifiers were trained:

Logistic Regression – Linear baseline

Decision Tree Classifier – Non-linear, interpretable model

Random Forest Classifier – Ensemble of decision trees

Support Vector Machine (SVM) – Optimal separating hyperplane

K-Nearest Neighbors (k-NN) – Distance-based classification

Gradient Boosting Classifier – Sequential boosting of weak learners

⚙️ Stage 4: Hyperparameter Tuning
Tool: RandomizedSearchCV

🏆 Best Model: Random Forest Tuning Results
Total Fits: 5 folds for each of 20 candidates, totaling 100 fits

Best Cross-Validated Accuracy: 0.9999

Best Parameters Found: [Optimized hyperparameters stored in model]

📊 Stage 5: Evaluation Metrics
Accuracy – Overall correct predictions

Classification Report – Precision, recall, F1-score per class

Confusion Matrix – Breakdown of prediction outcomes

📈 Stage 6: Model Performance Results
Model Comparison
Model Name	Accuracy
Random Forest Classifier	0.999915
Decision Tree Classifier	0.999894
Logistic Regression	0.999576
K-Nearest Neighbors	0.997010
Support Vector Machine (SVM)	0.997010
🔍 Final Model Deep Dive: Random Forest
After tuning, the Random Forest model was evaluated on the test set:

Accuracy: 0.9999151805593842

Classification Report
text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     23464
           1       1.00      1.00      1.00     23464

    accuracy                           1.00     46928
   macro avg       1.00      1.00      1.00     46928
weighted avg       1.00      1.00      1.00     46928
Confusion Matrix
The model misclassified only 4 legitimate URLs as phishing and made zero errors in identifying actual phishing URLs.

🏆 Stage 7: Conclusion
Based on the evaluation, the Random Forest Classifier achieved the highest accuracy (0.999915), indicating it is the most effective model for this dataset.

Key reasons for its performance:
Ability to handle both numerical and categorical features

Robustness against overfitting due to ensemble averaging

High interpretability via feature importance

Final Decision:
The Random Forest Classifier is selected as the final model for phishing URL detection. The trained model has been saved as random_forest_model.pkl for integration into a security pipeline for real-time threat detection.

🛠️ Tech Stack
Language: Python

Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

Tools: Jupyter Notebook

✅ Current Project Status
✔️ Model training and final evaluation completed

✔️ Final model saved as random_forest_model.pkl
