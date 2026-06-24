# 📉 Advanced Telco Customer Churn Prediction Engine

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## 📌 Project Overview
Customer churn is a critical financial drain for subscription-based telecom businesses. This project builds a production-grade, end-to-end Machine Learning pipeline designed to discover, optimize, and deploy predictive models to identify high-risk customer profiles. 

Rather than settling for standard binary outputs, this engine utilizes a dual-model framework optimized using randomized search algorithms, customized decision thresholds to maximize **Recall**, and an automated **Risk Tier Matrix** that maps mathematical probabilities directly into actionable business strategies.

---

## 🛠️ Tech Stack & Architecture
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning & Modeling:** Scikit-Learn (Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, Random Forest, AdaBoost)
* **Hyperparameter Optimization:** `RandomizedSearchCV` with parallel multi-core processing (`n_jobs=-1`)
* **Model Serialization & Deployment:** Joblib
* **Evaluation Metrics:** Classification Report, Precision-Recall Curve, F1-Score, Confusion Matrix

---

## 🧠 The Machine Learning Pipeline

### 1. Exploratory Data Analysis (EDA)
* Handled structural missing values (e.g., converting empty string spaces into `0` for the `TotalCharges` column of brand-new customers with a tenure of 0).
* Analyzed continuous feature distributions (`tenure`, `MonthlyCharges`) to detect non-linear boundaries across churn patterns.

### 2. Preprocessing & Feature Engineering
* **Categorical Encoding:** Leveraged One-Hot Encoding for multi-class strings (e.g., Internet Service variants) while avoiding the dummy variable trap through multicollinearity isolation.
* **Feature Scaling:** Implemented a robust `StandardScaler` fitted strictly on the training partition to safeguard against data leakage during cross-validation.
* **Alignment Matrix:** Built an automated alignment pipeline using `.reindex(fill_value=0)` to guarantee that single-row live inferences perfectly match the exact feature shape used during model training.

### 3. Model Exploration & Benchmarking
To find the most resilient architecture, five distinct algorithmic families were trained and benchmarked against one another:
* Baseline Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Trees
* **Random Forest** (Selected Top Performer)
* **AdaBoost** (Selected Top Performer)

### 4. Advanced Hyperparameter Tuning
The two most promising ensemble frameworks—**Random Forest** and **AdaBoost**—were subjected to an optimized `RandomizedSearchCV` pipeline over a complex hyperparameter distribution grid.
* Tuned architectural depth, estimator boundaries (`n_estimators`), and splitting criteria.
* Leveraged 3-fold stratified cross-validation over reduced, high-impact iteration cycles (`n_iter`) utilizing parallel background compute blocks to drastically reduce optimization latency while preserving convergence quality.

---

## 🚧 Key Challenges & Solutions

### Challenge 1: The "Accuracy Paradox" (Imbalanced Data)
Because the vast majority of telecom users do not cancel their service, raw baseline models achieved high structural accuracy (~82%) simply by predicting a blanket "No Churn." This resulted in a dangerously low Class 1 Recall, missing critical flight risks.
* **Solution:** Extracted raw soft probabilities via `.predict_proba()[:, 1]` rather than utilizing default hard cutoffs.
* **Result:** Tuned custom classification decision thresholds downward (e.g., 0.35) to favor maximum capture. This engineering shift dramatically pushed model Recall to catch high-probability churners before they left the platform.

### Challenge 2: Dual Model Verification
Relying on a single classification algorithm can build systemic architectural biases into a production engine. 
* **Solution:** Serialized both the optimized **Random Forest** (which learns deep, non-linear feature boundaries) and **AdaBoost** (which builds shallow, sequential learners targeting historical misclassifications).
* **Result:** The deployment engine reads inputs through both models simultaneously, allowing systems to evaluate risk profile divergences under different architectural lenses.

---

## 💼 Business Impact: End-to-End Operational Risk Tiers
The final serialized assets (`random_forest_model.pkl`, `adaboost_model.pkl`, and `scaler.pkl`) are bundled inside a unified operational script. The software combines predictions from both architectures, takes a conservative approach by evaluating the maximum predicted risk, and maps it directly to a strategic retention chart:

| Risk Percentage Range | Assessed Risk Tier | Business Assessment | Strategic Action Plan |
| :--- | :--- | :--- | :--- |
| **< 30%** | 🟢 **LOW RISK** | Safe, loyal profile. | Do not exhaust marketing budgets; target for premium upgrades. |
| **30% - 49%** | 🟡 **MODERATE RISK** | Wavering profile showing early churn indicators. | Dispatch automated nurture tracks or low-cost loyalty rewards. |
| **50% - 74%** | 🟠 **HIGH RISK** | Actively considering alternative service providers. | Trigger a high-priority 10-15% contract renewal discount. |
| **≥ 75%** | 🔴 **CRITICAL RISK** | High probability of imminent account cancellation. | Route to premium human retention desk; authorize maximum billing credits. |

---

## 🚀 How to Run Inference Locally
Once you clone the repository, you can pass individual partial customer attributes to the pipeline. The matrix automatically aligns missing columns, applies the pre-fitted scaling, runs both models, and returns your corporate action strategy:

```python
# Load the unified operational function
predict_churn_and_act({
    'tenure': 2, 
    'MonthlyCharges': 95.50, 
    'TotalCharges': 191.00,
    'PaperlessBilling_Yes': 1, 
    'PaymentMethod_Electronic check': 1
})
