# %%
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from loguru import logger
from sklift.metrics import qini_auc_score

sys.path.append(str(Path.cwd().parent / "src"))

from data.starbucks import load_data
from metrics.ranking import plot_uplift_curve
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
# %%
feature_cols = [col for col in train_df.columns if col.startswith("V")]

X_train = train_df[feature_cols].values
t_train = train_df["Promotion"].values
y_train = train_df["purchase"].values

# Train the uplift random forest model
uplift_rf = init_model(model_name="uplift_rf", control_name="No")
uplift_rf.fit(X=X_train, treatment=t_train, y=y_train)

# %%
# Evaluate model performance
X_test = test_df[feature_cols].values
test_df["uplift"] = uplift_rf.predict(X_test)
auqc = qini_auc_score(test_df["purchase"], test_df["uplift"], test_df["treatment"])
logger.info(f"Qini coefficient on test data: {auqc:.4f}")

plot_uplift_curve(test_df["purchase"], test_df["uplift"], test_df["treatment"])

# %%
# Examine feature importance
pd.Series(uplift_rf.feature_importances_, index=feature_cols).sort_values().plot(
    kind="barh", figsize=(12, 8)
)

# %%
# Plot uplift tree
from causalml.inference.tree import uplift_tree_plot
from IPython.display import Image

uplift_tree = uplift_rf.uplift_forest[0]
graph = uplift_tree_plot(uplift_tree.fitted_uplift_tree, feature_cols)
Image(graph.create_png())

# %%
# compute SHAP values
np.random.seed(42)

# Sample background
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

explainer = shap.PermutationExplainer(uplift_rf.predict, background)
shap_values = explainer(X_train[:500])
shap.summary_plot(shap_values, features=X_train[:500], feature_names=feature_cols)
# %%
