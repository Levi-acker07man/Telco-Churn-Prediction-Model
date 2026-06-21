# 📉 Telco Customer Churn Prediction Engine

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## 📌 Project Overview
Customer churn is a critical metric for subscription-based businesses. This project builds a complete, end-to-end Machine Learning pipeline to predict which customers are at the highest risk of leaving a telecommunications company. 

Beyond just raw accuracy, this model is engineered for **business value**, focusing heavily on maximizing **Recall** to catch at-risk customers before they cancel their service, and deploying a custom prediction engine that categorizes customers into actionable "Risk Tiers."

---

## 🛠️ Tech Stack & Architecture
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Logistic Regression, Random Forest, GridSearchCV)
* **Model Serialization:** Joblib
* **Evaluation Metrics:** Precision-Recall, F1-Score, Confusion Matrix

---

## 🧠 The Machine Learning Pipeline

### 1. Exploratory Data Analysis (EDA)
* Identified and handled hidden missing values (e.g., blank spaces in the `TotalCharges` column for brand-new customers).
* Analyzed the distribution of continuous variables (`tenure`, `MonthlyCharges`) against the target variable (`Churn`).

### 2. Data Preprocessing & Feature Engineering
* **Binary Encoding:** Converted text-based binary columns (e.g., "Yes" / "No") into machine-readable `1` and `0` integers.
* **One-Hot Encoding:** Transformed multi-category text features (like Internet Service type) into separate binary columns.
* **Multicollinearity Removal:** Dropped redundant "baseline" columns during encoding to prevent the dummy variable trap.
* **Feature Scaling:** Applied `StandardScaler` to continuous variables to optimize the Gradient Descent process, strictly fitting the scaler *only* on training data to prevent data leakage.

### 3. Model Training & Hyperparameter Tuning
* Built a foundational **Logistic Regression** model.
* Executed an exhaustive `GridSearchCV` using `StratifiedKFold` cross-validation (5 splits) to ensure robust performance across imbalanced datasets.
* Tuned the `C` parameter (regularization strength) and specific `solver` algorithms (`liblinear`, `lbfgs`).

---

## 🚧 Key Challenges & Solutions

### Challenge 1: The "Accuracy Paradox" (Imbalanced Data)
The initial dataset was highly imbalanced (most people do not churn). The raw model achieved 82% accuracy simply by guessing "Loyal" for almost everyone, resulting in a terrible Class 1 Recall of **59%** (missing nearly half the churners).
* **Solution:** Engineered a custom `class_weight` grid search to heavily penalize the model for missing Class 1. 
* **Result:** Successfully pushed the Class 1 Recall to **77%**, prioritizing the detection of high-risk customers over vanity accuracy.

### Challenge 2: The Precision-Recall Tradeoff
Forcing the model to be aggressive increased false positives (flagging loyal customers as risks). 
* **Solution:** Extracted exact mathematical probabilities using `.predict_proba()` instead of strict 1/0 predictions. 
* **Result:** Developed a custom tiering function to segment users based on probability thresholds rather than a binary output.

---

## 💼 Business Impact: The Risk Tier Engine
Instead of basic "Churn/No Churn" outputs, the serialized model is wrapped in a custom Python function that outputs specific marketing directives based on the probability calculation:

* 🔴 **CRITICAL (> 50%):** Immediate intervention required (e.g., Send 20% discount).
* 🟠 **HIGH RISK (36% - 50%):** Proactive retention call.
* 🟡 **MONITOR (16% - 35%):** Send satisfaction survey.
* 🟢 **SAFE (< 15%):** Loyal customer, no budget expenditure required.
