from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
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
