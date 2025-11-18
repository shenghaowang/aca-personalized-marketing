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
    plt.figure(figsize=(10, 7))
    frac = np.array(range(11)) / 100

    for i, model_name in enumerate(model_names):
        plt.plot(frac, avg[i], label=model_labels[model_name], color=colors[model_name])

        if ci_low is not None and ci_high is not None:
            plt.fill_between(
                frac, ci_low[i], ci_high[i], color=colors[model_name], alpha=0.2
            )

    plt.xlabel("Fraction of participants")
    plt.ylabel("Percentage decrease (%)")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
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
    plt.figure(figsize=(10, 7))
    sns.histplot(
        before,
        bins=50,
        alpha=0.5,
        label="Before collective action",
        color="blue",
        stat="density",
    )
    sns.histplot(
        after,
        bins=50,
        alpha=0.5,
        label="After collective action",
        color="orange",
        stat="density",
    )
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
