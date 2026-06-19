from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd

def compute_feature_importance(model, X_test, y_test, feature_names,
                               n_repeats=15, random_state=42):

    print("\n[EXPLAINABILITY] Computing permutation importance...")

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc"
    )

    sorted_idx = np.argsort(perm.importances_mean)[::-1]

    return pd.DataFrame({
        "feature": np.array(feature_names)[sorted_idx],
        "importance": perm.importances_mean[sorted_idx],
        "std": perm.importances_std[sorted_idx]
    })