import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def _print_distribution(name, values):
    classes, counts = np.unique(values, return_counts=True)
    distribution = dict(zip(classes.tolist(), counts.tolist()))
    print(f"{name}: {distribution}")


def _print_evaluation_diagnostics(model_name, y_test, y_pred):
    print(f"\n[{model_name}] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    _print_distribution(f"[{model_name}] Predicted class distribution", y_pred)
    _print_distribution(f"[{model_name}] True class distribution", y_test)

    if len(np.unique(y_pred)) == 1:
        print("WARNING: Model predicts only one class")


def evaluate_models(models, X_test, X_test_sc, y_test, config=None):
    config = config or {}
    threshold = config.get("evaluation", {}).get("prediction_threshold", 0.5)
    print(f"Using prediction threshold: {threshold}")

    rf_cal = models["rf_model"]
    lr_cal = models["lr_model"]

    y_proba_rf = rf_cal.predict_proba(X_test)[:, 1]
    y_proba_lr = lr_cal.predict_proba(X_test_sc)[:, 1]

    y_pred_rf = (y_proba_rf >= threshold).astype(int)
    y_pred_lr = (y_proba_lr >= threshold).astype(int)

    _print_evaluation_diagnostics("RF", y_test, y_pred_rf)
    _print_evaluation_diagnostics("LR", y_test, y_pred_lr)

    return {
        "rf_proba": y_proba_rf,
        "lr_proba": y_proba_lr,
        "rf_pred": y_pred_rf,
        "lr_pred": y_pred_lr,
        "rf_auc": roc_auc_score(y_test, y_proba_rf),
        "lr_auc": roc_auc_score(y_test, y_proba_lr),
        "rf_report": classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0),
        "lr_report": classification_report(y_test, y_pred_lr, output_dict=True, zero_division=0)
    }
