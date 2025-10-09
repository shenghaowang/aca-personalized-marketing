from pathlib import Path

import pandas as pd


def load_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df["Promotion"] = train_df["Promotion"].map({"Yes": 1, "No": 0})
    test_df["Promotion"] = test_df["Promotion"].map({"Yes": 1, "No": 0})

    train_df["label"] = train_df.apply(
        lambda row: 1 if row.Promotion == row.purchase else 0, axis=1
    )
    test_df["label"] = test_df.apply(
        lambda row: 1 if row.Promotion == row.purchase else 0, axis=1
    )

    return train_df, test_df
