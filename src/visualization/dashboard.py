import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pandas as pd

from sklearn.metrics import (
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    roc_auc_score,
    confusion_matrix
)


def create_dashboard(
    y_test,
    y_proba_lr,
    y_proba_rf,
    y_pred_rf,
    test_results,
    feature_importance_df,
    fairness_results: dict,
    output_path="reports/dashboard/risk_system_dashboard.png"
):
    """
    Generates full ML monitoring dashboard:
    - Model performance
    - Risk distribution
    - Feature importance
    - Fairness analysis
    """

    print("\n[VIZ] Generating dashboard...")

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Employee Attrition Risk Scoring System — Model Dashboard",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ---------------- ROC CURVE ----------------
    ax_roc = fig.add_subplot(gs[0, 0])

    RocCurveDisplay.from_predictions(
        y_test, y_proba_rf, ax=ax_roc,
        name=f"RF (AUC={roc_auc_score(y_test, y_proba_rf):.2f})"
    )

    RocCurveDisplay.from_predictions(
        y_test, y_proba_lr, ax=ax_roc,
        name=f"LR (AUC={roc_auc_score(y_test, y_proba_lr):.2f})"
    )

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax_roc.set_title("ROC Curve")

    # ---------------- CONFUSION MATRIX ----------------
    ax_cm = fig.add_subplot(gs[0, 1])

    cm = confusion_matrix(y_test, y_pred_rf)
    ConfusionMatrixDisplay(cm, display_labels=["Stays", "Leaves"]).plot(
        ax=ax_cm,
        colorbar=False
    )
    ax_cm.set_title("Confusion Matrix (RF)")

    # ---------------- RISK DISTRIBUTION ----------------
    ax_tier = fig.add_subplot(gs[0, 2])

    tier_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    tier_counts = (
        test_results["risk_tier"]
        .value_counts()
        .reindex(tier_order, fill_value=0)
    )

    ax_tier.bar(tier_order, tier_counts.values)
    ax_tier.set_title("Risk Tier Distribution")

    # ---------------- FEATURE IMPORTANCE ----------------
    ax_imp = fig.add_subplot(gs[1, :2])

    sorted_features = feature_importance_df["feature"].values
    importance = feature_importance_df["importance"].values
    std = feature_importance_df["std"].values

    top_n = min(12, len(sorted_features))

    ax_imp.barh(
        sorted_features[-top_n:],
        importance[-top_n:],
        xerr=std[-top_n:]
    )

    ax_imp.set_title("Feature Importance (Permutation)")
    ax_imp.set_xlabel("Importance")

    # ---------------- RISK SCORE DISTRIBUTION ----------------
    ax_dist = fig.add_subplot(gs[1, 2])

    ax_dist.hist(
        test_results[test_results["true_attrition"] == 0]["risk_score"],
        bins=20,
        alpha=0.7,
        label="Stays"
    )

    ax_dist.hist(
        test_results[test_results["true_attrition"] == 1]["risk_score"],
        bins=20,
        alpha=0.7,
        label="Leaves"
    )

    ax_dist.set_title("Risk Score Distribution")
    ax_dist.set_xlabel("Risk Score")
    ax_dist.legend()

    # ---------------- FAIRNESS ----------------
    ax_fair = fig.add_subplot(gs[2, :])

    all_fairness = pd.concat(fairness_results.values(), ignore_index=True)
    all_fairness = all_fairness.fillna(0)

    x = np.arange(len(all_fairness))
    width = 0.3

    ax_fair.bar(x - width, all_fairness["positive_rate"], width, label="Positive Rate")
    ax_fair.bar(x, all_fairness["tpr"], width, label="TPR (Recall)")
    ax_fair.bar(x + width, all_fairness["avg_risk"] / 100, width, label="Avg Risk")

    ax_fair.set_xticks(x)
    ax_fair.set_xticklabels(all_fairness["group"], rotation=20)
    ax_fair.set_title("Fairness Analysis — Group Comparison")
    ax_fair.legend()

    # ---------------- SAVE ----------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {output_path}")
