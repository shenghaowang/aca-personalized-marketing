from typing import List

import matplotlib.pyplot as plt
import numpy as np

model_labels = {"lgbm": "CT-LGBM", "mlp": "CT-MLP", "uplift_rf": "Uplift-RF"}
colors = {"lgbm": "red", "mlp": "blue", "uplift_rf": "orange"}


def plot_pct_decrease(
    avg: List[np.ndarray],
    ci_low: List[np.ndarray],
    ci_high: List[np.ndarray],
    model_names: List[str],
    title: str,
):
    plt.figure(figsize=(10, 7))
    frac = np.array(range(11)) / 100

    # Plot Qini coefficients
    for i, model_name in enumerate(model_names):
        plt.plot(frac, avg[i], label=model_labels[model_name], color=colors[model_name])
        plt.fill_between(
            frac, ci_low[i], ci_high[i], color=colors[model_name], alpha=0.2
        )

    plt.xlabel("Fraction of participants")
    plt.ylabel("Percentage decrease (%)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()
