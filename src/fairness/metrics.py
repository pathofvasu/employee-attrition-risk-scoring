import pandas as pd
import numpy as np

def compute_fairness_metrics(df_fairness, X_test_index, y_test,
                             y_pred, y_proba, group_cols):

    print("\n[FAIRNESS] Analyzing demographic group disparities...")

    df = df_fairness.loc[X_test_index].copy()
    df["predicted_prob"] = y_proba
    df["predicted_label"] = y_pred.astype(int)
    df["true_label"] = y_test.values.astype(int)

    def group_metrics(df, col):
        results = []

        for g in df[col].dropna().unique():
            sub = df[df[col] == g]

            if len(sub) < 5:
                continue

            true_pos = ((sub["true_label"] == 1) & (sub["predicted_label"] == 1)).sum()
            total_pos = sub["true_label"].sum()

            results.append({
                "group": f"{col}={g}",
                "count": len(sub),
                "positive_rate": sub["predicted_label"].mean(),
                "tpr": true_pos / max(total_pos, 1),
                "avg_risk": sub["predicted_prob"].mean()
            })

        return pd.DataFrame(results)

    return {col: group_metrics(df, col) for col in group_cols}