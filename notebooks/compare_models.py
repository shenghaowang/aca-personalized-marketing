# %%
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from sklift.metrics import qini_auc_score

sys.path.append(str(Path.cwd().parent / "src"))

from data.starbucks import load_data
from metrics.ranking import number_responses, uplift_curve
from model.model_type import init_model

# %%
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# %%
train_df, test_df = load_data(
    "../data/starbucks/training.csv", "../data/starbucks/test.csv"
)

logger.info(f"Training data shape: {train_df.shape}")
logger.info(f"Test data shape: {test_df.shape}")

train_df.head()

# %%
feature_cols = [col for col in train_df.columns if col.startswith("V")]

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values
y_train = train_df["label"].values

# %%
# Train the LGBM transformation models
ct_lgbm = init_model(model_name="lgbm", input_dim=len(feature_cols))
ct_lgbm.fit(X_train, y_train)
uplift_ct_lgbm = 2 * ct_lgbm.predict_proba(X_test)[:, 1] - 1

auqc = qini_auc_score(test_df["purchase"], uplift_ct_lgbm, test_df["treatment"])
logger.info(f"Qini coefficient on test data: {auqc:.4f}")

# %%
# Train the MLP class transformation model
ct_mlp = init_model(model_name="mlp", input_dim=len(feature_cols))
ct_mlp.fit(X_train, y_train)
uplift_ct_mlp = 2 * ct_mlp.predict_proba(X_test)[:, 1] - 1

auqc = qini_auc_score(test_df["purchase"], uplift_ct_mlp, test_df["treatment"])
logger.info(f"Qini coefficient on test data: {auqc:.4f}")

# %%
X_train = train_df[feature_cols].values
t_train = train_df["Promotion"].values
y_train = train_df["purchase"].values

# Train the uplift random forest model
uplift_rf = init_model(model_name="uplift_rf", control_name="No")
uplift_rf.fit(X=X_train, treatment=t_train, y=y_train)
# uplift_uplift_rf = uplift_rf.predict(X_test)
test_df["uplift"] = uplift_rf.predict(X_test)

auqc = qini_auc_score(test_df["purchase"], test_df["uplift"], test_df["treatment"])
logger.info(f"Qini coefficient on test data: {auqc:.4f}")

# %%
# Plot Qini curves
_, ax = plt.subplots(figsize=(10, 7))
xs, ys = uplift_curve(
    test_df["purchase"], uplift_ct_mlp, test_df["treatment"], n_nodes=None
)
ax.plot(xs, ys, label="CT-MLP", color="blue")
xs, ys = uplift_curve(
    test_df["purchase"], uplift_ct_lgbm, test_df["treatment"], n_nodes=None
)
ax.plot(xs, ys, label="CT-LGBM", color="red")
xs, ys = uplift_curve(
    test_df["purchase"], test_df["uplift"], test_df["treatment"], n_nodes=None
)
ax.plot(xs, ys, label="Uplift-RF", color="orange")

# random model
responses_target, rescaled_responses_control = number_responses(
    test_df["purchase"], test_df["treatment"]
)
incr_responses = responses_target - rescaled_responses_control
ax.plot(
    [0, len(test_df)],
    [0, incr_responses],
    label="Random",
    color="green",
    linestyle="--",
)

ax.set_xlabel("Number of individuals targeted")
ax.set_ylabel("Cumulative uplift")
ax.legend()
ax.grid(True)

# Export the figure
plt.tight_layout()
plt.savefig("uplift_curve.png")
# %%
