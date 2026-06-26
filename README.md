# 📉 Telco Customer Churn Prediction Engine

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-FF6F00?style=for-the-badge&logo=python&logoColor=white)

## 📌 Project Overview
Customer churn is a critical metric for subscription-based businesses. This project builds a complete, end-to-end Machine Learning pipeline to predict which customers are at the highest risk of leaving a telecommunications company.

Beyond just raw accuracy, this model is engineered for **business value**, focusing heavily on maximizing **Recall** to catch at-risk customers before they cancel their service, and deploying a custom prediction engine that categorizes customers into actionable "Risk Tiers."

---

## 🛠️ Tech Stack & Architecture
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, GridSearchCV)
* **Class Imbalance Handling:** imbalanced-learn (SMOTE)
* **Model Serialization:** Joblib
* **Evaluation Metrics:** Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

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

### 3. Class Imbalance Handling (SMOTE)
* The dataset was imbalanced — **5,174 Stayed vs 1,869 Churned** (73% vs 27%).
* Applied **SMOTE (Synthetic Minority Oversampling Technique)** on training data only to generate synthetic churn samples and balance the classes.
* SMOTE was applied strictly *after* train-test split and *after* scaling to prevent data leakage into the test set.

### 4. Model Training & Comparison
Trained and evaluated **5 classification models** to identify the best performer:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~82% | 0.69 | 0.77 | 0.73 | ~0.84 |
| Decision Tree | ~79% | 0.65 | 0.72 | 0.68 | ~0.79 |
| Random Forest | ~83% | 0.72 | 0.74 | 0.73 | ~0.86 |
| Gradient Boosting | ~84% | 0.74 | 0.73 | 0.73 | ~0.87 |
| AdaBoost | ~82% | 0.70 | 0.76 | 0.73 | ~0.85 |

> **Winner: Gradient Boosting** — highest overall balance of Precision, Recall and ROC-AUC.

### 5. Hyperparameter Tuning
* Executed an exhaustive `GridSearchCV` using `StratifiedKFold` cross-validation (5 splits) to ensure robust performance across imbalanced datasets.
* Tuned the `C` parameter (regularization strength) for Logistic Regression and `n_estimators`, `max_depth` for ensemble models.

---

## 🚧 Key Challenges & Solutions

### Challenge 1: The "Accuracy Paradox" (Imbalanced Data)
The initial dataset was highly imbalanced (most people do not churn). The raw model achieved 82% accuracy simply by guessing "Loyal" for almost everyone, resulting in a terrible Class 1 Recall of **59%** (missing nearly half the churners).
* **Solution 1:** Applied SMOTE oversampling to balance training data from 5174/1869 to equal classes.
* **Solution 2:** Engineered a custom `class_weight` grid search to heavily penalize the model for missing Class 1.
* **Result:** Successfully pushed the Class 1 Recall from **59% → 77%**, catching significantly more at-risk customers.

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

---

## 🚀 How to Use This Project

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Levi-acker07man/Telco-Churn-Prediction-Model.git
cd Telco-Churn-Prediction-Model
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Generate the Models
> ⚠️ Model `.pkl` files are **not included** in this repo due to GitHub's 100MB file size limit (Random Forest model = ~174MB). You must generate them locally by running the notebook.

```bash
# Open and run all cells in order
jupyter notebook Notebooks/Preprocessing.ipynb
```

This will automatically save the following files in the `Models/` folder:
- `random_forest_model.pkl`
- `adaboost_model.pkl`
- `scaler.pkl`

### Step 4 — Make a Prediction (Sample Code)

```python
import joblib
import pandas as pd

# Load models and scaler
rf_model  = joblib.load('Models/random_forest_model.pkl')
ada_model = joblib.load('Models/adaboost_model.pkl')
scaler    = joblib.load('Models/scaler.pkl')

expected_cols = rf_model.feature_names_in_

def predict_churn(customer_data):
    """
    Pass a dictionary of customer features.
    Only include features relevant to the customer — 
    missing columns are auto-filled with 0.
    """
    df = pd.DataFrame([customer_data]).reindex(columns=expected_cols, fill_value=0)

    # Scale numerical columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.transform(df[num_cols])

    rf_risk  = rf_model.predict_proba(df)[0, 1]
    ada_risk = ada_model.predict_proba(df)[0, 1]

    print(f"Random Forest Churn Risk : {rf_risk:.2%}")
    print(f"AdaBoost Churn Risk      : {ada_risk:.2%}")

    # Risk Tier
    avg_risk = (rf_risk + ada_risk) / 2
    if avg_risk > 0.50:
        print("🔴 CRITICAL — Send immediate retention offer (e.g. 20% discount)")
    elif avg_risk > 0.36:
        print("🟠 HIGH RISK — Schedule proactive retention call")
    elif avg_risk > 0.15:
        print("🟡 MONITOR — Send customer satisfaction survey")
    else:
        print("🟢 SAFE — Loyal customer, no action needed")


# ── Example: High-risk customer ──────────────────────────
predict_churn({
    'tenure': 2,
    'MonthlyCharges': 95.50,
    'TotalCharges': 191.00,
    'PaperlessBilling_Yes': 1,
    'PaymentMethod_Electronic check': 1,
    'Contract_One year': 0,
    'Contract_Two year': 0
})

# ── Example: Low-risk customer ───────────────────────────
predict_churn({
    'tenure': 48,
    'MonthlyCharges': 45.00,
    'TotalCharges': 2160.00,
    'Contract_Two year': 1,
    'PaymentMethod_Electronic check': 0
})
```

**Expected output for high-risk customer:**
```
Random Forest Churn Risk : 67.30%
AdaBoost Churn Risk      : 61.20%
🔴 CRITICAL — Send immediate retention offer (e.g. 20% discount)
```

**Expected output for low-risk customer:**
```
Random Forest Churn Risk : 8.40%
AdaBoost Churn Risk      : 11.20%
🟢 SAFE — Loyal customer, no action needed
```

---

## 📁 Project Structure
```
Telco-Churn-Prediction-Model/
│
├── Datasets/
│   └── Telco_cleaned.csv          # Preprocessed dataset
│
├── Models/                        # Generated after running notebook
│   ├── random_forest_model.pkl    # ⚠️ Not in repo — generate locally
│   ├── adaboost_model.pkl         # ⚠️ Not in repo — generate locally
│   └── scaler.pkl
│
├── Notebooks/
│   ├── EDA.ipynb                  # Exploratory Data Analysis
│   └── Preprocessing.ipynb        # Full ML pipeline + model training
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📦 Requirements
```
pandas
numpy
scikit-learn
imbalanced-learn
joblib
matplotlib
seaborn
jupyter
```

---

## 👤 Author
**Abhinav Parmar**  
B.Tech Mechanical Engineering | PDPM IIITDM Jabalpur  
[GitHub](https://github.com/Levi-acker07man) · [LinkedIn](https://linkedin.com/in/your-profile)
