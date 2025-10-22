import random
from typing import List, Union

import pandas as pd
from causalml.inference.tree import UpliftRandomForestClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from sklift.metrics import qini_auc_score

from model.neuralnet import NeuralNetClassifier


def experiment(
    model: Union[LGBMClassifier, NeuralNetClassifier, UpliftRandomForestClassifier],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    attack_attr: str,
    attack_val: int,
    new_val: int,
    frac: float = 0.1,
):
    train_df_modified = collective_action(
        df=train_df,
        attack_attr=attack_attr,
        attack_val=attack_val,
        new_val=new_val,
        frac=frac,
    )
    test_df_modified = collective_action(
        df=test_df,
        attack_attr=attack_attr,
        attack_val=attack_val,
        new_val=new_val,
        frac=frac,
    )

    X_train_modified = train_df_modified[feature_cols].values
    X_test_modified = test_df_modified[feature_cols].values

    # Fit model and predict uplift based on model type
    if isinstance(model, UpliftRandomForestClassifier):
        # Uplift Random Forest model
        t_train_modified = train_df_modified["Promotion"].values
        y_train_modified = train_df_modified["purchase"].values
        model.fit(X=X_train_modified, treatment=t_train_modified, y=y_train_modified)
        test_df_modified["uplift"] = model.predict(X_test_modified)
    else:
        # Class transformation models (LGBM, MLP)
        y_train_modified = train_df_modified["label"].values
        model.fit(X_train_modified, y_train_modified)
        test_df_modified["uplift"] = 2 * model.predict_proba(X_test_modified)[:, 1] - 1

    # Convert treatment to binary for metric calculation
    treatment_binary = (test_df_modified["Promotion"] == "Yes").astype(int)
    auqc = qini_auc_score(
        test_df_modified["purchase"],
        test_df_modified["uplift"],
        treatment_binary,
    )

    test_df_modified["rank"] = (
        test_df_modified["uplift"].rank(method="dense", ascending=False).astype(int)
    )
    test_df_modified["normalised_rank"] = (
        test_df_modified["rank"] / test_df_modified["rank"].max()
    )

    normalised_rank_df = pd.merge(
        test_df[["ID", attack_attr, "normalised_rank"]],
        test_df_modified[["ID", "normalised_rank", "aca_flag"]],
        on="ID",
        suffixes=["", "_modified"],
    )

    collective_df = normalised_rank_df[normalised_rank_df["aca_flag"] == 1]

    return {
        "qini_coeff": auqc,
        "num_participants": collective_df.shape[0],
        "avg_normalised_rank": collective_df["normalised_rank"].mean(),
        "median_normalised_rank": collective_df["normalised_rank"].median(),
        "avg_modified_normalised_rank": collective_df[
            "normalised_rank_modified"
        ].mean(),
        "median_modified_normalised_rank": collective_df[
            "normalised_rank_modified"
        ].median(),
    }


# Strategies: V6 = 4 -> 1, V1 = 3 -> 0
def collective_action(
    df: pd.DataFrame, attack_attr: str, attack_val: int, new_val: int, frac: float = 0.1
) -> pd.DataFrame:
    # Sample the data by ID
    collective_ids = df[df[attack_attr] == attack_val]["ID"].tolist()

    # random.seed(42)
    sampled_ids = random.sample(collective_ids, int(len(collective_ids) * frac))
    logger.info(
        f"Number of customers who participate in the collective action: {len(sampled_ids)}"
    )

    df_sampled = df[df["ID"].isin(sampled_ids)].copy()
    df_sampled.loc[:, "aca_flag"] = 1
    df_unsampled = df[~df["ID"].isin(sampled_ids)].copy()
    df_unsampled.loc[:, "aca_flag"] = 0

    logger.info(
        f"Sampled shape: {df_sampled.shape}, Unsampled shape: {df_unsampled.shape}"
    )

    df_sampled[attack_attr] = new_val
    return pd.concat([df_sampled, df_unsampled])
