from typing import List, Tuple

import numpy as np
import pandas as pd
import shap
import torch
from causalml.inference.tree import UpliftRandomForestClassifier
from lightgbm import LGBMClassifier

from model.neuralnet import NeuralNetClassifier


def compute_shap_values(model, X: np.ndarray, seed: int = 42):
    """
    Compute SHAP values for a given model with reproducible results.

    Args:
        model: Trained model (LGBM, MLP, or UpliftRF)
        X: Input features
        seed: Random seed for reproducibility

    Returns:
        Tuple of (shap_values, X_used, X_original)
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    if isinstance(model, LGBMClassifier):
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        X_used = X
        X_original = X

    elif isinstance(model, NeuralNetClassifier):
        X_scaled = model.scaler.transform(X)

        # Use deterministic k-means clustering for background
        # Note: shap.kmeans uses sklearn's KMeans which respects np.random.seed
        X_bg = shap.kmeans(X_scaled, 100).data

        # Sample from scaled data
        n_samples = min(1000, X_scaled.shape[0])
        sample_indices = np.random.choice(
            X_scaled.shape[0], size=n_samples, replace=False
        )
        sample_indices = np.sort(sample_indices)  # Sort for deterministic ordering

        X_sample = X_scaled[sample_indices]
        X_original = X[sample_indices]  # Get original unscaled features

        # Convert to torch tensors
        bg_t = torch.tensor(X_bg, dtype=torch.float32)
        sample_t = torch.tensor(X_sample, dtype=torch.float32)

        # Set model to eval mode for deterministic behavior
        model.model.eval()

        explainer = shap.GradientExplainer(model.model, bg_t)
        shap_values = explainer.shap_values(sample_t)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Ensure shap_values is 2D: (n_samples, n_features)
        shap_values = np.array(shap_values).squeeze()
        X_used = X_sample  # Normalized features for SHAP correlation

    elif isinstance(model, UpliftRandomForestClassifier):
        # Use deterministic background sampling
        n_background = min(100, X.shape[0])
        background_indices = np.random.choice(
            X.shape[0], size=n_background, replace=False
        )
        background_indices = np.sort(background_indices)
        background = X[background_indices]

        # PermutationExplainer has internal randomness - need to set seed in masker
        explainer = shap.PermutationExplainer(
            model.predict, background, seed=seed  # Control permutation randomness
        )

        # Use first N samples deterministically
        n_subset = min(500, X.shape[0])
        X_subset = X[:n_subset]
        shap_values = explainer(X_subset)
        X_used = X_subset
        X_original = X_subset

    else:
        raise ValueError(f"Unsupported model type for SHAP values: {type(model)}")

    return shap_values, X_used, X_original


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
    if correlation == 0:
        return "N/A (no correlation)", np.nan, np.nan

    # Check if feature is categorical
    if is_categorical_feature(feature_values):
        min_val = feature_values.min()
        max_val = feature_values.max()

        # Positive correlation: higher values lead to higher SHAP (positive impact)
        # Strategy: modify min to max to increase positive impact
        if correlation > 0:
            strategy = f"Modify {feature_name} from {min_val} to {max_val}"
            return strategy, min_val, max_val

        # Negative correlation: higher values lead to lower/negative SHAP
        # Strategy: modify max to min to reduce negative impact
        else:
            strategy = f"Modify {feature_name} from {max_val} to {min_val}"
            return strategy, max_val, min_val

    else:
        # For continuous features, compute offset based on percentiles
        p10 = np.percentile(feature_values, 10)
        p90 = np.percentile(feature_values, 90)

        if correlation > 0:
            p20 = np.percentile(feature_values, 20)
            strategy = (
                f"Increase {feature_name} by {p90 - p10:.2f} for values below {p20:.2f}"
            )

            return strategy, p10, p90

        else:
            p80 = np.percentile(feature_values, 80)
            strategy = (
                f"Decrease {feature_name} by {p90 - p10:.2f} for values above {p80:.2f}"
            )

            return strategy, p90, p10


def report_feature_contribution(
    shap_values, X: np.ndarray, feature_cols: List[str], X_original: np.ndarray = None
):
    """
    Report feature contributions and propose collective actions.

    Args:
        shap_values: SHAP values computed on (possibly normalized) features
        X: Feature values used for computing SHAP (may be normalized for MLP)
        feature_cols: List of feature names
        X_original: Original unnormalized feature values for strategy proposals.
                   If None, uses X for both correlation and strategy proposals.
    """
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

    # Use original features for strategy proposals if provided
    X_for_strategy = X_original if X_original is not None else X_data

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
        # Use X_data (normalized for MLP) for correlation calculation
        corr = np.corrcoef(X_data[:, i], shap_array[:, i])[0, 1]
        correlations.append(corr)

        # Propose collective action strategy using original feature values
        strategy, from_val, to_val = propose_collective_action(
            X_for_strategy[:, i], corr, feature_cols[i]
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
        .sort_values("correlation", ascending=False)
        .reset_index(drop=True)
    )

    feature_impact_df["rank"] = range(1, len(feature_impact_df) + 1)

    return feature_impact_df
