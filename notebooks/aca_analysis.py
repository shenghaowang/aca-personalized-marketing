# %%
import matplotlib.pyplot as plt
import numpy as np


# %%
def plot_aca(
    q1: np.ndarray, i1: np.ndarray, q2: np.ndarray, i2: np.ndarray, title: str
):
    plt.figure(figsize=(10, 7))
    frac = np.array([0.01, 0.03, 0.05, 0.07, 0.1])
    q1_lower, q1_upper = q1 - i1, q1 + i1
    q2_lower, q2_upper = q2 - i2, q2 + i2

    # Plot Qini coefficients
    plt.plot(frac, q1, label="LGBM", color="blue")
    plt.fill_between(frac, q1_lower, q1_upper, color="blue", alpha=0.2)

    plt.plot(frac, q2, label="MLP", color="red")
    plt.fill_between(frac, q2_lower, q2_upper, color="red", alpha=0.2)

    plt.xlabel("Fraction of participants")
    plt.ylabel("Percentage decrease (%)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


# %%
# Strategy 1: Change V6 = 4 to 1 - Qini Coefficient Decrease
plot_aca(
    q1=np.array([12.00, 14.47, 16.88, 15.17, 14.87]),
    i1=np.array([2.36, 2.39, 2.81, 2.77, 2.68]),
    q2=np.array([28.28, 30.8, 34.46, 43.43, 29.74]),
    i2=np.array([10.41, 9.26, 10.74, 10.26, 10.51]),
    title="Strategy 1: Change V6 = 4 to 1 - Qini Coefficient Decrease",
)

# %%
# Strategy 1: Change V6 = 4 to 1 - Average Normalised Rank Decrease
plot_aca(
    q1=np.array([17.95, 20.26, 23.00, 23.08, 25.56]),
    i1=np.array([1.51, 1.37, 1.46, 1.66, 2.29]),
    q2=np.array([-9.91, 2.9, -0.19, -2.46, 0.56]),
    i2=np.array([3.45, 3.95, 3.79, 3.42, 3.49]),
    title="Strategy 1: Change V6 = 4 to 1 - Average Normalised Rank Decrease",
)

# %%
# Strategy 2: Change V6 = 4 to 1 - Qini Coefficient Decrease
plot_aca(
    q1=np.array([10.50, 15.31, 16.79, 16.17, 16.36]),
    i1=np.array([2.08, 1.97, 2.14, 2.09, 2.21]),
    q2=np.array([42.28, 42.63, 38.68, 49.63, 13.04]),
    i2=np.array([9.62, 10.40, 10.15, 9.98, 10.88]),
    title="Strategy 2: Change V6 = 4 to 1 - Qini Coefficient Decrease",
)

# %%
# Strategy 2: Change V6 = 4 to 1 - Average Normalised Rank Decrease
plot_aca(
    q1=np.array([18.41, 20.62, 21.4, 23.15, 24.10]),
    i1=np.array([1.15, 0.99, 1.19, 1.15, 1.13]),
    q2=np.array([-1.61, -4.17, 2.79, -3.26, 1.48]),
    i2=np.array([4.77, 4.38, 4.63, 4.2, 4.34]),
    title="Strategy 2: Change V6 = 4 to 1 - Average Normalised Rank Decrease",
)
