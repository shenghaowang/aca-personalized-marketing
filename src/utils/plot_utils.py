from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

model_labels = {"lgbm": "CT-LGBM", "mlp": "CT-MLP", "uplift_rf": "Uplift-RF"}
colors = {"lgbm": "red", "mlp": "blue", "uplift_rf": "orange"}


def plot_pct_decrease(
    model_names: List[str],
    title: str,
    output_path: Path,
    avg: List[np.ndarray],
    ci_low: List[np.ndarray] = None,
    ci_high: List[np.ndarray] = None,
):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    frac = np.array(range(11)) / 100

    for i, model_name in enumerate(model_names):
        ax.plot(frac, avg[i], label=model_labels[model_name], color=colors[model_name])

        if ci_low is not None and ci_high is not None:
            ax.fill_between(
                frac, ci_low[i], ci_high[i], color=colors[model_name], alpha=0.2
            )

    ax.set_xlabel("Fraction of participants")
    ax.set_ylabel("Percentage decrease (%)")
    ax.set_title(title)
    ax.grid(True)

    # Legend with white background
    legend = ax.legend()
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_alpha(1.0)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, facecolor="white")
    plt.close()


def compare_distributions(
    before: np.ndarray,
    after: np.ndarray,
    output_path: Path,
    xlabel: str = "Value",
    title: str = "Distribution Before and After Collective Action",
):
    """Compare distributions before and after collective action.

    Args:
        before: Array of values before collective action
        after: Array of values after collective action
        output_path: Path to save the figure
        xlabel: Label for x-axis (default: "Value")
        title: Title for the plot (default: "Distribution Before and After Collective Action")
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sns.histplot(
        before,
        bins=50,
        alpha=0.5,
        label="Before collective action",
        color="blue",
        stat="density",
        ax=ax,
    )
    sns.histplot(
        after,
        bins=50,
        alpha=0.5,
        label="After collective action",
        color="orange",
        stat="density",
        ax=ax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Legend with white background
    legend = ax.legend()
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_alpha(1.0)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, facecolor="white")
    plt.close()


def plot_qini_with_collective_distribution(
    test_df: pd.DataFrame,
    normalised_rank_df: pd.DataFrame,
    target_col: str,
    qini_score_before: float,
    qini_score_after: float,
    model_name: str,
    dataset_name: str,
    output_path: Path,
):
    """Plot Qini curves with collective user distribution overlay.

    Args:
        test_df: Test dataframe with treatment and target columns
        normalised_rank_df: Dataframe with uplift, uplift_modified, and aca_flag columns
        target_col: Name of the target column
        qini_score_before: Qini coefficient before collective action
        qini_score_after: Qini coefficient after collective action
        model_name: Name of the model (for labels)
        dataset_name: Name of the dataset (for title)
        output_path: Path to save the figure
    """
    from eval.ranking import number_responses, uplift_curve

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    # Plot Qini curves on primary y-axis
    xs, ys = uplift_curve(
        test_df[target_col],
        normalised_rank_df["uplift"],
        test_df["treatment"],
        n_nodes=None,
    )
    ax1.plot(
        xs,
        ys,
        label=f"Before collective action: {qini_score_before:.4f}",
        color="blue",
        linewidth=2,
    )

    xs, ys = uplift_curve(
        test_df[target_col],
        normalised_rank_df["uplift_modified"],
        test_df["treatment"],
        n_nodes=None,
    )
    ax1.plot(
        xs,
        ys,
        label=f"After collective action: {qini_score_after:.4f}",
        color="orange",
        linewidth=2,
    )

    # Add random model baseline
    responses_target, rescaled_responses_control = number_responses(
        test_df[target_col], test_df["treatment"]
    )
    incr_responses = responses_target - rescaled_responses_control
    ax1.plot(
        [0, len(test_df)],
        [0, incr_responses],
        label="Random",
        color="green",
        linestyle="--",
        linewidth=2,
    )

    # Formatting for primary y-axis
    ax1.set_xlabel("Number of individuals targeted")
    ax1.set_ylabel("Cumulative uplift", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True, alpha=0.3)

    # Create secondary y-axis for collective user distribution
    ax2 = ax1.twinx()
    ax2.set_facecolor("white")

    # Get collective users data
    collective_df = normalised_rank_df[normalised_rank_df["aca_flag"] == 1].copy()

    # Get ranking positions in the full dataset
    full_df_before = normalised_rank_df.sort_values(
        "uplift", ascending=False
    ).reset_index(drop=True)
    full_df_after = normalised_rank_df.sort_values(
        "uplift_modified", ascending=False
    ).reset_index(drop=True)

    # Get x-positions for collective users in the full ranking
    x_positions_before = []
    x_positions_after = []
    for _, row in collective_df.iterrows():
        rank_position_before = full_df_before[full_df_before["ID"] == row["ID"]].index[
            0
        ]
        x_positions_before.append(rank_position_before)

        rank_position_after = full_df_after[full_df_after["ID"] == row["ID"]].index[0]
        x_positions_after.append(rank_position_after)

    # Plot density histograms for collective users
    if len(x_positions_before) > 0 or len(x_positions_after) > 0:
        # Use more bins to make bars thinner since collective users are a small proportion
        num_bins = 200

        if len(x_positions_before) > 0:
            ax2.hist(
                x_positions_before,
                bins=num_bins,
                alpha=0.4,
                color="blue",
                label="Collective users (before)",
                range=(0, len(test_df)),
            )

        if len(x_positions_after) > 0:
            ax2.hist(
                x_positions_after,
                bins=num_bins,
                alpha=0.4,
                color="orange",
                label="Collective users (after)",
                range=(0, len(test_df)),
            )

    # Formatting for secondary y-axis
    ax2.set_ylabel("Number of Collective Users", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_alpha(1.0)

    ax1.set_title(
        f"Qini Curves and Ranking Distribution of {model_labels[model_name]} for {dataset_name} Dataset"
    )

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
