from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)

from src.utils.config_loader import project_path


def analyze_thresholds(y_true, y_proba, model_name, output_dir=None):

    thresholds = np.arange(0.10, 0.91, 0.01)

    results = []

    for threshold in thresholds:

        y_pred = (y_proba >= threshold).astype(int)

        results.append({
            "model": model_name,
            "threshold": round(threshold, 2),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0)
        })


    df = pd.DataFrame(results)

    best = df.loc[df["f1"].idxmax()]

    if output_dir is None:
        output_dir = project_path("reports/model_selection")
    elif isinstance(output_dir, str):
        output_dir = project_path(output_dir)

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path or str")

    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_")


    df.to_csv(
        output_dir / f"{safe_name}_thresholds.csv",
        index=False
    )


    print("\n==============================")
    print(model_name)
    print("==============================")
    print("Best Threshold:", best["threshold"])
    print("F1:", round(best["f1"],3))
    print("Precision:", round(best["precision"],3))
    print("Recall:", round(best["recall"],3))


    return best, df