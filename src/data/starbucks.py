from pathlib import Path

import pandas as pd


def load_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Create binary treatment variable
    train_df["treatment"] = train_df["Promotion"].apply(
        lambda x: 1 if x == "Yes" else 0
    )
    test_df["treatment"] = test_df["Promotion"].apply(lambda x: 1 if x == "Yes" else 0)

    train_df["label"] = (
        ((train_df["treatment"] == 1) & (train_df["purchase"] == 1))
        | ((train_df["treatment"] == 0) & (train_df["purchase"] == 0))
    ).astype(int)

    test_df["label"] = (
        ((test_df["treatment"] == 1) & (test_df["purchase"] == 1))
        | ((test_df["treatment"] == 0) & (test_df["purchase"] == 0))
    ).astype(int)

    return train_df, test_df
