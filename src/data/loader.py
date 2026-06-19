import os

import pandas as pd

from src.data.generator import generate_synthetic_attrition
from src.utils.config_loader import project_path


def _resolve_project_path(path):
    if os.path.isabs(path):
        return path
    return str(project_path(path))


def load_data(config=None, logger=None):
    config = config or {}
    data_config = config.get("data", {})
    paths_config = config.get("paths", {})

    raw_config = data_config.get("raw", "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    path = _resolve_project_path(raw_config)
    synthetic_fallback = data_config.get("synthetic_fallback", True)
    random_state = data_config.get("random_state", 42)
    synthetic_path = _resolve_project_path(paths_config.get("synthetic_data", "data/raw/attrition_synthetic.csv"))

    # 1. Try real dataset
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
        message = f"Data shape: {df.shape} | Attrition rate: {df['Attrition'].mean():.1%}"
        if logger:
            logger.info("Real dataset loaded")
            logger.info(message)
        else:
            print("[DATA] Real dataset loaded")
            print(f"[DATA] {message}")
        return df

    if not synthetic_fallback:
        raise FileNotFoundError(f"Dataset not found: {path}")

    # 2. Fallback synthetic
    if logger:
        logger.info("Real dataset not found - generating synthetic data")
    else:
        print("[WARNING] Real dataset not found - generating synthetic data")

    df = generate_synthetic_attrition(seed=random_state)

    os.makedirs(os.path.dirname(synthetic_path), exist_ok=True)
    df.to_csv(synthetic_path, index=False)

    message = f"Data shape: {df.shape} | Attrition rate: {df['Attrition'].mean():.1%}"
    if logger:
        logger.info(message)
    else:
        print(f"[DATA] {message}")

    return df
