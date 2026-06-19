# Employee Attrition Risk Scoring System

An explainable and fairness-aware Machine Learning system that predicts employee attrition risk, converts prediction probabilities into actionable risk scores, analyzes model fairness, and generates HR decision-support reports.

The project is designed as an end-to-end ML engineering pipeline with:

* Configuration-driven execution
* Automated training and evaluation
* Model comparison and selection
* Threshold optimization
* Explainability analysis
* Fairness auditing
* Dashboard generation

---

# Problem Statement

Employee attrition creates significant operational and financial challenges for organizations.

Traditional prediction systems only answer:

> "Will an employee leave or not?"

This system provides more useful HR intelligence:

* Who is at risk?
* How high is the risk?
* Why is the employee considered high risk?
* Are predictions fair across demographic groups?

---

# System Architecture

```
IBM HR Analytics Dataset
          |
          ↓
Data Loading
(Config-driven + Synthetic fallback)
          |
          ↓
Feature Preprocessing
(Label Encoding + Scaling)
          |
          ↓
Model Training
 ┌───────────────────────┐
 │ Logistic Regression   │
 │ Random Forest         │
 └───────────────────────┘
          |
          ↓
Model Evaluation
(ROC-AUC + Metrics)
          |
          ↓
Threshold Analysis
          |
          ↓
Final Model Selection
(Logistic Regression)
          |
          ↓
Risk Score Generation
(0-100%)
          |
          ↓
Explainability
(Permutation Importance)
          |
          ↓
Fairness Analysis
(Gender + Age Groups)
          |
          ↓
Dashboard Generation
```

---

# Dataset

## IBM HR Analytics Employee Attrition Dataset

Source:

Kaggle - IBM HR Analytics Employee Attrition & Performance

Dataset:

* Employees: 1470
* Features: 35
* Target variable: Attrition
* Attrition rate: 16.1%

Dataset location:

```
data/
└── raw/
    └── WA_Fn-UseC_-HR-Employee-Attrition.csv
```

If the real dataset is unavailable, the pipeline automatically generates synthetic data for development testing.

---

# Machine Learning Models

The system trains and compares two classification models:

| Model               | Purpose                      |
| ------------------- | ---------------------------- |
| Logistic Regression | Interpretable baseline model |
| Random Forest       | Non-linear ensemble model    |

## Validation Results

| Model               | CV ROC-AUC |
| ------------------- | ---------: |
| Logistic Regression |      0.800 |
| Random Forest       |      0.775 |

---

# Threshold Optimization and Final Model Selection

Instead of using a default classification threshold, the system performs threshold analysis.

Generated reports:

```
reports/
└── model_selection/
    ├── logistic_regression_thresholds.csv
    ├── random_forest_thresholds.csv
    └── threshold_results.csv
```

Final selected model:

```
Model:
Logistic Regression

Optimal Threshold:
0.37
```

Selection was based on balancing:

* F1-score
* Precision
* Recall

---

# Risk Scoring System

The model probability is converted into a human-readable risk score.

Example:

```
Prediction Probability = 0.72

Risk Score = 72/100
```

Risk categories:

| Score  | Risk Level | Recommended Action     |
| ------ | ---------- | ---------------------- |
| 75-100 | Critical   | Immediate intervention |
| 55-74  | High       | Retention discussion   |
| 35-54  | Medium     | Monitor employee       |
| 0-34   | Low        | Normal engagement      |

---

# Explainability

The system uses permutation importance to understand feature influence.

The goal is not only prediction but explanation:

> "Why is this employee considered at risk?"

Important factors analyzed include:

* Overtime
* Job satisfaction
* Income level
* Career growth
* Work-life balance

---

# Fairness Analysis

The system evaluates whether predictions show demographic disparities.

Analysis groups:

## Gender

* Male
* Female

## Age Groups

* 18-30
* 31-40
* 41+

Generated reports:

```
reports/
└── fairness/
    ├── gender_fairness.csv
    └── age_fairness.csv
```

---

# Configuration Driven Pipeline

All important parameters are controlled using YAML files:

```
config/

├── config.yaml
        Data and training settings

├── model_config.yaml
        Model hyperparameters

├── thresholds.yaml
        Risk categories

└── paths.yaml
        Output locations
```

This avoids hardcoded values and makes the pipeline easier to maintain.

---

# Project Structure

```
MAX_PROJECT/

├── artifacts/
│   ├── models/
│   │   ├── logistic_regression.pkl
│   │   └── random_forest.pkl
│   │
│   ├── encoders/
│   │   └── encoders.pkl
│   │
│   ├── scaler/
│   │   └── scaler.pkl
│   │
│   └── model_selection/
│       └── final_model.json
│
├── config/
│   ├── config.yaml
│   ├── model_config.yaml
│   ├── paths.yaml
│   └── thresholds.yaml
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── reports/
│   ├── dashboard/
│   │   └── risk_system_dashboard.png
│   │
│   ├── fairness/
│   │   ├── age_fairness.csv
│   │   └── gender_fairness.csv
│   │
│   └── model_selection/
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
├── requirements.txt
└── run_pipeline.py
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>

cd MAX_PROJECT
```

Create virtual environment:

### Windows

```powershell
python -m venv venv

venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python -m venv venv

source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

Execute:

```bash
python run_pipeline.py
```

The complete workflow runs automatically:

```
Data Loading
        ↓
Preprocessing
        ↓
Training
        ↓
Evaluation
        ↓
Threshold Analysis
        ↓
Risk Scoring
        ↓
Explainability
        ↓
Fairness Analysis
        ↓
Dashboard Generation
```

---

# Generated Outputs

## Models

```
artifacts/models/

logistic_regression.pkl
random_forest.pkl
```

## Preprocessing Objects

```
artifacts/

encoders.pkl
scaler.pkl
```

## Reports

```
reports/

dashboard/
fairness/
model_selection/
```

---

# Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* YAML Configuration
* Joblib

---

# Future Improvements

Possible extensions:

* FastAPI deployment
* Real-time employee risk prediction API
* Model monitoring
* Automated retraining pipeline
* Cloud deployment
* Advanced explainability using SHAP

---

Built as an explainable ML decision-support system for employee retention analytics.
