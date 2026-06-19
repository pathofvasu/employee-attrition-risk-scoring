---
# 🧠 Employee Attrition Risk Intelligence System

A production-style, explainable and fairness-aware Machine Learning system that predicts employee attrition risk, converts predictions into actionable HR risk scores, performs bias auditing, and generates decision-ready analytics dashboards.

This is not just a model — it is an end-to-end ML engineering pipeline.

---

## 🚨 Problem Statement

Employee attrition causes massive financial and operational losses in organizations.

Most ML systems stop at:

> “Will this employee leave?”

This system answers deeper questions:

* Who is likely to leave?
* How severe is the risk?
* Why is this employee at risk?
* Is the model fair across groups?
* Which probability threshold is optimal for decision-making?

---

## 🏗️ System Architecture

```
IBM HR Dataset
↓
Config-Driven Pipeline
↓
Preprocessing (Encoding + Scaling)
↓
Model Training
├── Logistic Regression
└── Random Forest
↓
Model Evaluation (ROC-AUC)
↓
Threshold Optimization (Business-driven)
↓
Final Model Selection
↓
Risk Scoring Engine (0–100)
↓
Explainability Layer
↓
Fairness Audit
↓
Dashboard Generation
```

---

## 📊 Dataset

* IBM HR Analytics Employee Attrition & Performance

* 1470 employees

* 35 features

* ~16.1% attrition rate

### Path

```
data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

If missing, synthetic data is automatically generated.

---

## 🤖 Machine Learning Models

| Model               | Role                   |
| ------------------- | ---------------------- |
| Logistic Regression | Interpretable baseline |
| Random Forest       | Non-linear ensemble    |

### Cross-Validation Performance

| Model               | ROC-AUC |
| ------------------- | ------- |
| Logistic Regression | ~0.800  |
| Random Forest       | ~0.775  |

---

## 🎯 Threshold Optimization & Model Selection

Instead of using a fixed 0.5 threshold, this system performs **data-driven threshold tuning**.

### Outputs

```
reports/model_selection/
├── logistic_regression_thresholds.csv
├── random_forest_thresholds.csv
└── threshold_results.csv
```

### Final Selected Model

```
Model: Logistic Regression
Optimal Threshold: 0.37
```

### Selection Criteria

* F1-score optimization
* Precision–Recall tradeoff
* Business interpretability

---

## ⚖️ Risk Scoring Engine

Predicted probabilities are converted into HR-ready risk scores (0–100).

### Risk Levels

| Score Range | Tier     | HR Action              |
| ----------- | -------- | ---------------------- |
| 75–100      | Critical | Immediate intervention |
| 55–74       | High     | Retention discussion   |
| 35–54       | Medium   | Monitoring             |
| 0–34        | Low      | No action              |

---

## 🧠 Explainability Layer

Uses permutation-based feature importance to explain predictions.

### Key Drivers

* OverTime
* Job Satisfaction
* Monthly Income
* Years Since Promotion
* Work-Life Balance

This ensures **interpretability for HR stakeholders**, not a black-box model.

---

## ⚖️ Fairness & Bias Audit

Evaluates demographic parity across:

* Gender
* Age groups

### Outputs

```
reports/fairness/
├── gender_fairness.csv
└── age_fairness.csv
```

### Metrics

* Demographic parity difference
* Positive rate gap

---

## 📊 Dashboard

Auto-generated HR analytics dashboard:

```
reports/dashboard/risk_system_dashboard.png
```

Includes:

* Risk distribution
* Model performance comparison
* Feature importance
* Fairness summary

---

## ⚙️ Configuration-Driven Design

Everything is controlled via YAML configs:

```
config/
├── config.yaml
├── model_config.yaml
├── thresholds.yaml
└── paths.yaml
```

No hardcoded ML logic → fully reproducible pipeline.

---

## 📁 Project Structure

```
MAX_PROJECT/
├── artifacts/
├── reports/
├── config/
├── data/
├── src/
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

```bash
git clone <repo-url>
cd MAX_PROJECT
pip install -r requirements.txt
```

---

## ▶️ Run Pipeline

```bash
python run_pipeline.py
```

Pipeline flow:

```
Load → Preprocess → Train → Evaluate → Threshold Tuning →
Model Selection → Risk Scoring → Explainability → Fairness → Dashboard
```

---

## 📦 Outputs

### Models

```
artifacts/models/
```

### Preprocessing

```
artifacts/encoders/
artifacts/scaler/
```

### Reports

```
reports/
├── dashboard/
├── fairness/
└── model_selection/
```

---

## 🧰 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* YAML configuration system

---

## 🏁 Project Status

* ✔ End-to-end ML pipeline
* ✔ Explainability layer
* ✔ Fairness audit system
* ✔ Threshold optimization
* ✔ Production-style architecture

---

## 🔮 Future Improvements

* FastAPI deployment
* Real-time prediction API
* SHAP explanations
* Cloud deployment (AWS/Azure)
* CI/CD retraining pipeline

---
