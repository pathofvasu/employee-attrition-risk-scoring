# 🧠 Employee Attrition Risk Intelligence System

A production-style, explainable Machine Learning system that predicts employee attrition risk, converts predictions into actionable HR risk scores, performs bias/fairness auditing, and generates decision-ready analytics dashboards.

This project is designed as a **full ML engineering pipeline**, not just a model training script.

---

# 🚨 Problem Statement

Employee attrition costs organizations significant time and financial loss.

Most ML systems only answer:

> “Will this employee leave?”

This system goes further and answers:

- Who is likely to leave?
- How severe is the risk?
- Why is the employee at risk?
- Is the model fair across demographic groups?
- Which probability threshold gives the best business performance?

---

# 🏗️ System Architecture

```

IBM HR Dataset
↓
Config-Driven Data Loading
↓
Preprocessing (Encoding + Scaling)
↓
Model Training
├── Logistic Regression
└── Random Forest
↓
Model Evaluation (ROC-AUC)
↓
Threshold Optimization (Business-tuned)
↓
Final Model Selection
↓
Risk Scoring Engine (0–100)
↓
Explainability Layer (Feature Importance)
↓
Fairness Audit (Demographic Parity)
↓
Dashboard Generation (HR Insights)

```

---

# 📊 Dataset

**IBM HR Analytics Employee Attrition Dataset (Kaggle)**

- 👥 1470 employees
- 📊 35 features
- ⚠️ Attrition rate: ~16.1%

📂 Path:
```

data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv

```

If missing, the system automatically generates a synthetic dataset for testing.

---

# 🤖 Machine Learning Models

| Model                | Role                        |
|---------------------|----------------------------|
| Logistic Regression  | Interpretable baseline     |
| Random Forest        | Non-linear ensemble model  |

### Cross-Validation Performance

| Model                | ROC-AUC |
|---------------------|--------:|
| Logistic Regression  | 0.800   |
| Random Forest        | 0.775   |

---

# 🎯 Threshold Optimization & Model Selection

Instead of using a fixed 0.5 threshold, this system performs **data-driven threshold tuning**.

### Generated Artifacts

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

- F1 Score optimization
- Precision–Recall tradeoff
- Business interpretability

---

# ⚖️ Risk Scoring Engine

The model outputs probabilities which are converted into HR-ready risk scores.

### Example

```

Prediction Probability → Risk Score
0.72 → 72 / 100

```

### Risk Levels

| Score Range | Risk Level | HR Action |
|------------|------------|----------|
| 75–100      | Critical   | Immediate intervention |
| 55–74       | High       | Retention discussion |
| 35–54       | Medium     | Monitoring required |
| 0–34        | Low        | No action needed |

---

# 🧠 Explainability Layer

Uses permutation importance to explain model behavior.

### Key Drivers of Attrition

- Overtime
- Job Satisfaction
- Monthly Income
- Years Since Promotion
- Work-Life Balance

This ensures the system is **interpretable for HR decision-makers**, not a black box.

---

# ⚖️ Fairness & Bias Audit

Evaluates model fairness across sensitive groups:

- Gender
- Age Groups

### Outputs

```

reports/fairness/
├── gender_fairness.csv
└── age_fairness.csv

```

### Metrics Checked

- Demographic parity
- Positive prediction rate gap

---

# 📊 Dashboard Preview

A full HR analytics dashboard is automatically generated:

```

reports/dashboard/risk_system_dashboard.png

```

Includes:

- Risk distribution overview
- Model comparison metrics
- Feature importance analysis
- Fairness audit summary

---

# ⚙️ Configuration-Driven Design

All system behavior is controlled via YAML configs:

```

config/
├── config.yaml        # Data + training settings
├── model_config.yaml  # Model hyperparameters
├── thresholds.yaml    # Risk tier logic
└── paths.yaml         # Output paths

```

No hardcoded parameters → fully reproducible pipeline.

---

# 📁 Project Structure

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

# 🚀 Installation

```bash
git clone <your-repo-url>
cd MAX_PROJECT
pip install -r requirements.txt
````

---

# ▶️ Run Pipeline

```bash
python run_pipeline.py
```

Pipeline execution:

```
Load Data → Preprocess → Train → Evaluate → Optimize Threshold →
Select Model → Risk Scoring → Explainability → Fairness → Dashboard
```

---

# 📦 Outputs

### Models

```
artifacts/models/
```

### Preprocessing Objects

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

# 🧰 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* YAML-based configuration
* Joblib serialization

---

# 🏁 Project Status

✔ End-to-end ML pipeline
✔ Explainable AI system
✔ Fairness-aware modeling
✔ Config-driven architecture
✔ Production-style structure

---

# 📌 GitHub Release v1.0.0

### Features

* Full ML pipeline automation
* Threshold-based model selection
* Explainability engine
* Fairness auditing system
* HR-ready risk scoring

### Outputs

* Dashboard visualization
* Model selection reports
* Fairness reports
* Serialized models

---

# 🔮 Future Improvements

* FastAPI deployment
* Real-time prediction API
* SHAP-based explanations
* Cloud deployment (AWS / Azure)
* CI/CD retraining pipeline

```


