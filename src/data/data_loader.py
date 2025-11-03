from typing import List, Tuple

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklift.datasets import fetch_hillstrom


def load_starbucks(
    data_cfg: DictConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    train_df = pd.read_csv(data_cfg.train_path)
    test_df = pd.read_csv(data_cfg.test_path)

    # Create binary treatment variable
    train_df["treatment"] = train_df[data_cfg.treatment_col].apply(
        lambda x: 1 if x == "Yes" else 0
    )
    test_df["treatment"] = test_df[data_cfg.treatment_col].apply(
        lambda x: 1 if x == "Yes" else 0
    )

    # Create binary label for class transformation models
    train_df["label"] = (
        ((train_df["treatment"] == 1) & (train_df[data_cfg.target_col] == 1))
        | ((train_df["treatment"] == 0) & (train_df[data_cfg.target_col] == 0))
    ).astype(int)
    test_df["label"] = (
        ((test_df["treatment"] == 1) & (test_df[data_cfg.target_col] == 1))
        | ((test_df["treatment"] == 0) & (test_df[data_cfg.target_col] == 0))
    ).astype(int)

    feature_cols = [col for col in train_df.columns if col.startswith("V")]

    return train_df, test_df, feature_cols


def load_hillstrom(
    data_cfg: DictConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    ds = fetch_hillstrom(target_col="visit")
    df = pd.concat([ds.data, ds.target, ds.treatment], axis=1)

    # Exclude mens' marketing data
    df = df[df["segment"] != "Mens E-Mail"]

    df["history_segment"] = df["history_segment"].apply(historic_segment_transform)
    df = pd.get_dummies(
        df, columns=["zip_code", "channel"], prefix=["zip_code", "channel"], dtype=int
    )

    # Create binary treatment variable
    df["treatment"] = df[data_cfg.treatment_col].map(
        {"Womens E-Mail": 1, "No E-Mail": 0}
    )

    # Create binary label for class transformation models
    df["label"] = (
        ((df["treatment"] == 1) & (df[data_cfg.target_col] == 1))
        | ((df["treatment"] == 0) & (df[data_cfg.target_col] == 0))
    ).astype(int)

    # Split into train and test sets
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df[data_cfg.treatment_col]
    )
    feature_cols = [
        col
        for col in df.columns
        if col
        not in (data_cfg.treatment_col, data_cfg.target_col, "treatment", "label")
    ]

    return train_df, test_df, feature_cols


def historic_segment_transform(payment: str) -> int:
    match payment:
        case "1) $0 - $100":
            return 50
        case "2) $100 - $200":
            return 150
        case "3) $200 - $350":
            return 275
        case "4) $350 - $500":
            return 425
        case "5) $500 - $750":
            return 575
        case "5) $750 - $1000":
            return 825
        case _:
            return 1000
