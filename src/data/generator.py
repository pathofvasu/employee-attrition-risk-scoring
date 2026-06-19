import numpy as np
import pandas as pd

def generate_synthetic_attrition(n=800, seed=42):

    np.random.seed(seed)

    age = np.random.randint(22, 60, n)
    gender = np.random.choice(["Male", "Female"], n)
    department = np.random.choice(["Sales", "Research", "HR"], n, p=[0.4, 0.45, 0.15])

    job_level = np.random.randint(1, 6, n)
    monthly_income = (job_level * 1200 + np.random.normal(0, 400, n)).clip(1000, 20000).astype(int)

    years_at_company = np.random.randint(0, 20, n)
    years_since_promotion = np.random.randint(0, 15, n)

    job_satisfaction = np.random.randint(1, 5, n)
    work_life_balance = np.random.randint(1, 5, n)

    overtime = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])

    distance_from_home = np.random.randint(1, 30, n)

    attrition_score = (
        -0.4 * job_satisfaction
        -0.3 * work_life_balance
        +0.5 * (overtime == "Yes").astype(int)
        -0.2 * (monthly_income / 5000)
        +0.3 * (years_since_promotion / 5)
        -0.2 * (years_at_company / 10)
        +0.2 * (distance_from_home / 15)
        + np.random.normal(0, 0.5, n)
    )

    prob = 1 / (1 + np.exp(-attrition_score))
    attrition = (np.random.rand(n) < prob).astype(int)

    return pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Department": department,
        "JobLevel": job_level,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company,
        "YearsSinceLastPromotion": years_since_promotion,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "OverTime": overtime,
        "DistanceFromHome": distance_from_home,
        "Attrition": attrition
    })