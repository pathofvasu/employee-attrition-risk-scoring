from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_data
from src.explainability.importance import compute_feature_importance
from src.fairness.metrics import compute_fairness_metrics
from src.features.preprocessing import encode_features, scale_features
from src.models.evaluate import evaluate_models
from src.models.train import train_models
from src.scoring.risk_engine import generate_risk_table
from src.models.threshold_analysis import analyze_thresholds
from src.utils.config_loader import load_config, project_path
from src.utils.logger import get_logger
from src.visualization.dashboard import create_dashboard


def _ensure_directories(config):
    paths = config.get("paths", {})
    for key in (
        "artifact_models",
        "artifact_encoders",
        "artifact_scaler",
        "artifact_model_selection",
        "reports_model_selection",
        "reports_fairness",
        "reports_dashboard",
    ):
        project_path(paths[key]).mkdir(parents=True, exist_ok=True)


def _save_artifacts(models, encoders, scaler, config):
    paths = config["paths"]
    joblib.dump(models["rf_model"], project_path(paths["artifact_models"]) / "random_forest.pkl")
    joblib.dump(models["lr_model"], project_path(paths["artifact_models"]) / "logistic_regression.pkl")
    joblib.dump(encoders, project_path(paths["artifact_encoders"]) / "encoders.pkl")
    joblib.dump(scaler, project_path(paths["artifact_scaler"]) / "scaler.pkl")


def _save_fairness_reports(fairness_results, config):
    fairness_path = project_path(config["paths"]["reports_fairness"])
    for group_name, report in fairness_results.items():
        output_name = "age_fairness.csv" if group_name == "AgeGroup" else f"{group_name.lower()}_fairness.csv"
        report.to_csv(fairness_path / output_name, index=False)


def main():
    config = load_config()
    logger = get_logger()
    _ensure_directories(config)

    print("\n" + "=" * 60)
    print("EMPLOYEE ATTRITION RISK SCORING SYSTEM")
    print("=" * 60)

    # ---------------- DATA ----------------
    df = load_data(config=config, logger=logger)
    logger.info("Data loaded")

    target = config.get("features", {}).get("target", "Attrition")
    data_config = config.get("data", {})

    df_fairness = df[["Age", "Gender", target]].copy()
    df_fairness["AgeGroup"] = pd.cut(
        df_fairness["Age"],
        bins=[18, 30, 40, 65],
        labels=["18-30", "31-40", "41+"]
    )

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_config.get("test_size", 0.2),
        random_state=data_config.get("random_state", 42),
        stratify=y
    )

    # ---------------- FEATURES ----------------
    X_train_enc, X_test_enc, encoders = encode_features(X_train, X_test)
    X_train_sc, X_test_sc, scaler = scale_features(X_train_enc, X_test_enc)

    feature_names = X_train_enc.columns.tolist()

    # ---------------- TRAIN ----------------
    logger.info("Training started")
    models = train_models(X_train_sc, X_train_enc, y_train, config=config)
    _save_artifacts(models, encoders, scaler, config)
    logger.info("Training completed")

    # ---------------- EVALUATE ----------------
    eval_results = evaluate_models(
        models,
        X_test_enc,
        X_test_sc,
        y_test,
        config=config
    )
    logger.info("Evaluation completed")
    # ---------------- THRESHOLD ANALYSIS ----------------

    lr_best, lr_thresholds = analyze_thresholds(
        y_test,
        eval_results["lr_proba"],
        "Logistic Regression",
        output_dir=config["paths"]["reports_model_selection"]
    )

    rf_best, rf_thresholds = analyze_thresholds(
        y_test,
        eval_results["rf_proba"],
        "Random Forest",
        output_dir=config["paths"]["reports_model_selection"]
    )

    # Save threshold comparison report
    model_selection_dir = project_path(config["paths"]["reports_model_selection"])
    model_selection_dir.mkdir(parents=True, exist_ok=True)

    threshold_results = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Best Threshold": float(round(lr_best["threshold"], 2)),
            "F1": float(round(lr_best["f1"], 3)),
            "Precision": float(round(lr_best["precision"], 3)),
            "Recall": float(round(lr_best["recall"], 3))
        },
        {
            "Model": "Random Forest",
            "Best Threshold": float(round(rf_best["threshold"], 2)),
            "F1": float(round(rf_best["f1"], 3)),
            "Precision": float(round(rf_best["precision"], 3)),
            "Recall": float(round(rf_best["recall"], 3))
        }
    ])
    threshold_results.to_csv(model_selection_dir / "threshold_results.csv", index=False)

    # Final Model Selection

    if rf_best["f1"] > lr_best["f1"]:
        final_model = "Random Forest"
        selection_metrics = rf_best
    else:
        final_model = "Logistic Regression"
        selection_metrics = lr_best

    final_threshold = float(round(selection_metrics["threshold"], 2))
    final_selection = {
        "model": final_model,
        "threshold": final_threshold,
        "selection_metric": "F1",
        "f1_score": float(round(selection_metrics["f1"], 3)),
        "precision": float(round(selection_metrics["precision"], 3)),
        "recall": float(round(selection_metrics["recall"], 3))
    }

    print("\n==============================")
    print("FINAL MODEL")
    print("==============================")
    print("Model:", final_model)
    print("Threshold:", final_threshold)

    artifacts_model_selection_dir = project_path(config["paths"]["artifact_model_selection"])
    artifacts_model_selection_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_model_selection_dir / "final_model.json", "w", encoding="utf-8") as output_file:
        json.dump(final_selection, output_file, indent=2)

    # ---------------- RISK ENGINE ----------------
    if final_model == "Logistic Regression":
        final_probability = eval_results["lr_proba"]
    else:
        final_probability = eval_results["rf_proba"]

    test_results = generate_risk_table(
        X_test_enc,
        y_test,
        final_probability
    )
    logger.info("Risk scoring completed")

    # ---------------- FEATURE IMPORTANCE ----------------
    feature_importance = compute_feature_importance(
        models["rf_model"],
        X_test_enc,
        y_test,
        feature_names,
        random_state=data_config.get("random_state", 42)
    )

    # ---------------- FAIRNESS ----------------
    fairness_results = compute_fairness_metrics(
        df_fairness,
        X_test_enc.index,
        y_test,
        eval_results["rf_pred"],
        eval_results["rf_proba"],
        group_cols=["Gender", "AgeGroup"]
    )
    _save_fairness_reports(fairness_results, config)
    logger.info("Fairness completed")

    # ---------------- DASHBOARD ----------------
    create_dashboard(
        y_test=y_test,
        y_proba_lr=eval_results["lr_proba"],
        y_proba_rf=eval_results["rf_proba"],
        y_pred_rf=eval_results["rf_pred"],
        test_results=test_results,
        feature_importance_df=feature_importance,
        fairness_results=fairness_results,
        output_path=str(project_path(config["paths"]["dashboard"]))
    )
    logger.info("Dashboard generated")

    print("\n[DONE] Pipeline executed successfully.")


if __name__ == "__main__":
    main()
