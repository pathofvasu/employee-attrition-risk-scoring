from sklearn.model_selection import train_test_split


def split_data(X, y, config=None):
    config = config or {}
    data_config = config.get("data", {})

    return train_test_split(
        X,
        y,
        test_size=data_config.get("test_size", 0.2),
        random_state=data_config.get("random_state", 42),
        stratify=y
    )
