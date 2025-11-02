from typing import List, Tuple

import numpy as np
import pandas as pd
import shap
import torch
from causalml.inference.tree import UpliftRandomForestClassifier
from lightgbm import LGBMClassifier

from model.neuralnet import NeuralNetClassifier


def compute_shap_values(model, X: np.ndarray):
    if isinstance(model, LGBMClassifier):
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        X_used = X

    elif isinstance(model, NeuralNetClassifier):
        X_scaled = model.scaler.transform(X)
        X_bg = shap.kmeans(X_scaled, 100).data
        X_sample = shap.utils.sample(X_scaled, 1000, 42)

        # Convert to torch tensors
        bg_t = torch.tensor(X_bg, dtype=torch.float32)
        sample_t = torch.tensor(X_sample, dtype=torch.float32)
        explainer = shap.GradientExplainer(model.model, bg_t)
        shap_values = explainer.shap_values(sample_t)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Ensure shap_values is 2D: (n_samples, n_features)
        shap_values = np.array(shap_values).squeeze()
        X_used = X_sample

    elif isinstance(model, UpliftRandomForestClassifier):
        background = X[np.random.choice(X.shape[0], 100, replace=False)]

        explainer = shap.PermutationExplainer(model.predict, background)
        X_subset = X[:500]
        shap_values = explainer(X_subset)
        X_used = X_subset

    else:
        raise ValueError(f"Unsupported model type for SHAP values: {type(model)}")

    return shap_values, X_used


def is_categorical_feature(feature_values: np.ndarray, threshold: int = 10) -> bool:
    """
    Determine if a feature is categorical based on the number of unique values.

    Args:
        feature_values: Array of feature values
        threshold: Maximum number of unique values to consider categorical

    Returns:
        True if feature is categorical, False if continuous
    """
    unique_values = np.unique(feature_values)
    return len(unique_values) <= threshold


def propose_collective_action(
    feature_values: np.ndarray, correlation: float, feature_name: str
) -> Tuple[str, float, float]:
    """
    Propose collective action strategy based on correlation and feature type.

    Args:
        feature_values: Array of feature values
        correlation: Pearson correlation between feature and SHAP values
        feature_name: Name of the feature

    Returns:
        Tuple of (strategy_description, from_value, to_value)
    """
    # Check if feature is categorical
    if not is_categorical_feature(feature_values):
        return "N/A (continuous feature)", np.nan, np.nan

    min_val = feature_values.min()
    max_val = feature_values.max()

    # Positive correlation: higher values lead to higher SHAP (positive impact)
    # Strategy: modify min to max to increase positive impact
    if correlation > 0:
        strategy = f"Modify {feature_name} from {min_val} to {max_val}"
        return strategy, min_val, max_val

    # Negative correlation: higher values lead to lower/negative SHAP
    # Strategy: modify max to min to reduce negative impact
    elif correlation < 0:
        strategy = f"Modify {feature_name} from {max_val} to {min_val}"
        return strategy, max_val, min_val

    else:
        return "N/A (no correlation)", np.nan, np.nan


def report_feature_contribution(shap_values, X: np.ndarray, feature_cols: List[str]):
    # Handle SHAP Explanation object
    if hasattr(shap_values, "values"):
        shap_array = shap_values.values
        # Get feature data from Explanation object if available
        if hasattr(shap_values, "data"):
            X_data = shap_values.data
        else:
            X_data = X
    else:
        shap_array = np.array(shap_values)
        X_data = X

    # Mean SHAP value = directional impact
    mean_shap = shap_array.mean(axis=0)

    # Mean absolute SHAP = feature importance (more accurate)
    mean_abs_shap = np.abs(shap_array).mean(axis=0)

    # Compute correlation between feature values and SHAP values
    # and propose collective action strategies
    correlations = []
    strategies = []
    from_values = []
    to_values = []

    for i in range(len(feature_cols)):
        # Pearson correlation between feature column and its SHAP values
        corr = np.corrcoef(X_data[:, i], shap_array[:, i])[0, 1]
        correlations.append(corr)

        # Propose collective action strategy
        strategy, from_val, to_val = propose_collective_action(
            X_data[:, i], corr, feature_cols[i]
        )
        strategies.append(strategy)
        from_values.append(from_val)
        to_values.append(to_val)

    feature_impact_df = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "mean_shap": mean_shap,
                "importance": mean_abs_shap,
                "correlation": correlations,
                "direction": [
                    "Positive" if x > 0 else "Negative" for x in correlations
                ],
                "proposed_strategy": strategies,
                "from_value": from_values,
                "to_value": to_values,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    feature_impact_df["rank"] = range(1, len(feature_impact_df) + 1)

    return feature_impact_df
