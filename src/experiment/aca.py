import random
from dataclasses import dataclass
from typing import Dict, List, Union

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns
from causalml.inference.tree import UpliftRandomForestClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from omegaconf import DictConfig
from scipy import stats
from sklift.metrics import qini_auc_score

from model.neuralnet import NeuralNetClassifier
from model.trainer import train_and_predict


@dataclass
class CollectiveAction:
    collective_criterion: Dict[str, str]
    attacks: List[Dict[str, str]]
    frac: float

    # Derived attributes
    feature: str = None
    operator: str = None
    threshold: float = None
    attack_features: List[str] = None

    def __post_init__(self):
        """Compute derived attributes from the base attributes."""
        # Extract feature name, operator, and threshold from collective_criterion
        self.feature, eligible_criterion = list(self.collective_criterion.items())[0]
        self.operator = eligible_criterion[0]
        self.threshold = float(eligible_criterion[1:])
        self.attack_features = []
        for attack in self.attacks:
            self.attack_features.extend(attack.keys())


def experiment(
    model: Union[LGBMClassifier, NeuralNetClassifier, UpliftRandomForestClassifier],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    data_cfg: DictConfig,
    action: CollectiveAction,
    seed: int = 42,
):
    """
    Run a single ACA experiment with given eligibility criteria and attack mappings.

    Args:
        model: Uplift model
        train_df: Training dataframe
        test_df: Test dataframe
        feature_cols: List of feature column names
        model_name: Name of the model
        data_cfg: Data configuration
        collective_criterion: Eligibility criteria for participation
        attack_mappings: Feature value modifications to apply
        frac: Fraction of eligible customers to participate
    """
    train_df_modified = apply_action(
        df=train_df,
        action=action,
        seed=seed,
    )
    test_df_modified = apply_action(
        df=test_df,
        action=action,
        seed=seed,
    )

    # Retrain model on perturbed data and predict uplift
    test_df_modified = train_and_predict(
        model=model,
        train_df=train_df_modified,
        test_df=test_df_modified,
        feature_cols=feature_cols,
        model_name=model_name,
        data_cfg=data_cfg,
    )

    # Calculate metric
    auqc = qini_auc_score(
        test_df_modified[data_cfg.target_col],
        test_df_modified["uplift"],
        test_df_modified["treatment"],
    )

    test_df_modified["rank"] = (
        test_df_modified["uplift"].rank(method="dense", ascending=False).astype(int)
    )
    test_df_modified["normalised_rank"] = (
        test_df_modified["rank"] / test_df_modified["rank"].max()
    )

    # Include attack features in the merge to track original values
    merge_cols = ["ID"] + action.attack_features + ["normalised_rank"]
    normalised_rank_df = pd.merge(
        test_df[merge_cols],
        test_df_modified[["ID", "normalised_rank", "aca_flag"]],
        on="ID",
        suffixes=["", "_modified"],
    )

    collective_df = normalised_rank_df[normalised_rank_df["aca_flag"] == 1]

    # Perform paired t-test
    ci_low, ci_high = paired_t_test(
        before=collective_df["normalised_rank"].values,
        after=collective_df["normalised_rank_modified"].values,
    )

    # plt.figure(figsize=(10, 7))
    # sns.histplot(
    #     data=collective_df,
    #     x='normalised_rank',
    #     bins=50, alpha=0.5, label='Before collective action', color='blue')
    # sns.histplot(
    #     data=collective_df,
    #     x='normalised_rank_modified',
    #     bins=50, alpha=0.5, label='After collective action', color='orange')
    # plt.legend()
    # plt.title('Normalised Rank Distribution Before and After Collective Action')
    # plt.show()

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
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def apply_action(
    df: pd.DataFrame,
    action: CollectiveAction,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Apply collective action by modifying feature values for eligible customers.

    Args:
        df: Input dataframe
        collective_criterion: Dict defining eligibility (who can participate).
                             Format: {feature_name: required_value}
                             Example: {"V4": 1} means customers with V4=1
        attack_mappings: List of dicts defining modifications (what to change).
                        Each dict has feature name as key and new value.
                        Example: [{"V4": 2}] means change V4 to 2
        frac: Fraction of eligible customers to participate

    Returns:
        Modified dataframe with 'aca_flag' column
    """

    # Apply operator to build eligibility mask
    if action.operator == "=":
        eligible_mask = df[action.feature] == action.threshold

    elif action.operator == ">":
        eligible_mask = df[action.feature] > action.threshold

    elif action.operator == "<":
        eligible_mask = df[action.feature] < action.threshold

    else:
        raise ValueError(
            f"Unsupported operator: {action.operator}. Use '=', '>', or '<'."
        )

    collective_ids = df[eligible_mask]["ID"].tolist()

    random.seed(seed)
    sampled_ids = random.sample(collective_ids, int(len(collective_ids) * action.frac))
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

    # Apply all feature value modifications specified in attack mappings
    for attack in action.attacks:
        for feature, attack_strategy in attack.items():
            operator, val = attack_strategy[0], float(attack_strategy[1:])
            if operator == "=":
                new_val = val

            elif operator == "+":
                new_val = df_sampled[feature] + val

            elif operator == "-":
                new_val = df_sampled[feature] - val

            else:
                raise ValueError(
                    f"Unsupported attack operator: {operator}. Use '=', '+', or '-'."
                )
            df_sampled.loc[:, feature] = new_val

    return pd.concat([df_sampled, df_unsampled])


def paired_t_test(before: np.ndarray, after: np.ndarray, confidence: float = 0.95):
    D = before - after
    n = len(D)
    Dbar, sD = D.mean(), D.std(ddof=1)
    SE = sD / np.sqrt(n)

    ci_low, ci_high = stats.t.interval(confidence, df=n - 1, loc=Dbar, scale=SE)
    return ci_low, ci_high
