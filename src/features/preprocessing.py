import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_features(X_train, X_test):

    encoders = {}
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    for col in X_train.select_dtypes(include="object").columns:

        le = LabelEncoder()
        X_train_enc[col] = le.fit_transform(X_train[col].astype(str))

        # SAFE handling unseen labels
        X_test_enc[col] = X_test[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

        encoders[col] = le

    return X_train_enc, X_test_enc, encoders


def scale_features(X_train, X_test):

    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return X_train_sc, X_test_sc, scaler