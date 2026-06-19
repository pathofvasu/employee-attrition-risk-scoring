from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def train_models(X_train_sc, X_train, y_train, config=None):
    config = config or {}
    lr_config = config.get("logistic_regression", config.get("model", {}).get("logistic_regression", {}))
    rf_config = config.get("random_forest", config.get("model", {}).get("random_forest", {}))
    training_config = config.get("training", {})
    random_state = config.get("data", {}).get("random_state", 42)
    calibration_cv = training_config.get("calibration_cv", 3)
    cv_folds = training_config.get("cv_folds", 5)

    # Logistic Regression
    lr = LogisticRegression(
        max_iter=lr_config.get("max_iter", 1000),
        random_state=random_state,
        class_weight=lr_config.get("class_weight", "balanced"),
        C=lr_config.get("C", 0.5)
    )

    lr_cal = CalibratedClassifierCV(lr, cv=calibration_cv)
    lr_cal.fit(X_train_sc, y_train)

    lr_cv = cross_val_score(lr, X_train_sc, y_train, cv=cv_folds, scoring="roc_auc")
    print(f"\n[LR] CV AUC: {lr_cv.mean():.3f} +/- {lr_cv.std():.3f}")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=rf_config.get("n_estimators", 150),
        max_depth=rf_config.get("max_depth", 8),
        random_state=random_state,
        class_weight=rf_config.get("class_weight", "balanced"),
        min_samples_leaf=rf_config.get("min_samples_leaf", 5)
    )

    rf_cal = CalibratedClassifierCV(rf, cv=calibration_cv)
    rf_cal.fit(X_train, y_train)

    rf_cv = cross_val_score(rf, X_train, y_train, cv=cv_folds, scoring="roc_auc")
    print(f"[RF] CV AUC: {rf_cv.mean():.3f} +/- {rf_cv.std():.3f}")

    return {
        "lr_model": lr_cal,
        "rf_model": rf_cal,
    }
