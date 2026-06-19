# Employee Attrition Risk Scoring System

An explainable, fairness-aware machine learning system that predicts employee attrition risk, converts probabilities into actionable risk scores, performs threshold-based model selection, and generates HR decision-support dashboards.

---

# Problem Statement

Traditional HR analytics answers only:

> Will an employee leave?

This system extends it into decision intelligence:

- Who is at risk?
- How severe is the risk?
- Why is the employee at risk?
- Is the model fair across demographics?
- What is the optimal decision threshold?

---

# System Architecture

```

IBM HR Dataset
↓
Data Loading (config-driven)
↓
Preprocessing (Encoding + Scaling)
↓
Model Training
├── Logistic Regression
└── Random Forest
↓
Model Evaluation (ROC-AUC)
↓
Threshold Optimization
↓
Final Model Selection
↓
Risk Scoring (0–100)
↓
Explainability (Permutation Importance)
↓
Fairness Audit (Gender, Age)
↓
Dashboard Generation

```

---

# Dataset

**IBM HR Analytics Employee Attrition Dataset**

- 1470 employees
- 35 features
- Attrition rate: 16.1%

📁 Path:
```

data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv

```

If missing → synthetic dataset is automatically generated.

---

# Models

| Model               | Role                    |
|--------------------|------------------------|
| Logistic Regression | Interpretable baseline |
| Random Forest       | Non-linear ensemble    |

## Cross Validation Performance

| Model               | ROC-AUC |
|--------------------|--------:|
| Logistic Regression | 0.800   |
| Random Forest       | 0.775   |

---

# Threshold Optimization & Model Selection

Instead of using a fixed threshold (0.5), the system performs full threshold optimization.

### Outputs

```

reports/model_selection/
├── logistic_regression_thresholds.csv
├── random_forest_thresholds.csv
└── threshold_results.csv

```

### Final Model

- **Model:** Logistic Regression  
- **Optimal Threshold:** 0.37  

Selection is based on:
- F1 Score
- Precision
- Recall

---

# Risk Scoring

Probability → Business Risk Score

```

0.72 → 72 / 100 Risk Score

```

## Risk Levels

| Score | Level     | Action |
|------:|----------|--------|
| 75–100 | Critical  | Immediate intervention |
| 55–74  | High      | Retention discussion |
| 35–54  | Medium    | Monitor employee |
| 0–34   | Low       | No action required |

---

# Explainability

Permutation importance identifies key drivers of attrition:

- Overtime
- Job Satisfaction
- Monthly Income
- Years Since Promotion
- Work-Life Balance

---

# Fairness Analysis

Evaluates model bias across:

- Gender
- Age Groups

### Outputs

```

reports/fairness/
├── gender_fairness.csv
└── age_fairness.csv

```

### Metrics

- Demographic parity
- Positive rate gap

---

# Dashboard Preview

Generated visualization:

```

reports/dashboard/risk_system_dashboard.png

```

Includes:
- Risk distribution
- Model comparison
- Feature importance
- Fairness summary

---

# Configuration System

All behavior is controlled via YAML configs:

```

config/
├── config.yaml        # data & training
├── model_config.yaml  # model hyperparameters
├── thresholds.yaml    # risk logic
└── paths.yaml         # output paths

```

No hardcoded parameters.

---

# Project Structure

```

MAX_PROJECT/
├── artifacts/
│   ├── models/
│   ├── encoders/
│   ├── scaler/
│   └── model_selection/
│
├── reports/
│   ├── dashboard/
│   ├── fairness/
│   └── model_selection/
│
├── config/
├── data/
│   └── raw/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── scoring/
│   ├── explainability/
│   ├── fairness/
│   ├── visualization/
│   └── utils/
│
├── run_pipeline.py
├── requirements.txt
└── README.md

````

---

# Installation

```bash
git clone <your-repo-url>
cd MAX_PROJECT
pip install -r requirements.txt
````

---

# Run Pipeline

```bash
python run_pipeline.py
```

Execution flow:

```
Data → Train → Evaluate → Threshold → Risk → Explain → Fairness → Dashboard
```

---

# Generated Outputs

### Models

```
artifacts/models/
```

### Preprocessing

```
artifacts/encoders.pkl
artifacts/scaler.pkl
```

### Reports

```
reports/
├── dashboard/
├── fairness/
└── model_selection/
```

---

# Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* YAML
* Joblib

---

# GitHub Release (v1.0.0)

### Features

* End-to-end ML pipeline
* Threshold optimization
* Explainable predictions
* Fairness auditing
* Config-driven architecture

### Outputs

* Risk scoring dashboard
* Model selection reports
* Fairness reports
* Serialized ML artifacts

### Status

✔ Production-ready ML pipeline
✔ Fully reproducible
✔ Config-driven design

---

# Future Improvements

* FastAPI deployment
* Real-time prediction API
* SHAP-based explainability
* Cloud deployment (AWS / Azure)
* Continuous training pipeline

```

---

