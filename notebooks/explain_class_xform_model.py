# %%
import random
import sys
from pathlib import Path

import numpy as np
import shap
import torch
from loguru import logger

from data.starbucks import load_data
from model.model_type import init_model

sys.path.append(str(Path.cwd().parent / "src"))

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

# Train the original class transformation model
X_train = train_df[feature_cols].values
y_train = train_df["label"].values

# %%
ct_lgbm = init_model(model_name="lgbm", input_dim=len(feature_cols))
ct_lgbm.fit(X_train, y_train)

# %%
# compute SHAP values
explainer = shap.Explainer(ct_lgbm)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, features=X_train, feature_names=feature_cols)

# %%
ct_mlp = init_model(model_name="mlp", input_dim=len(feature_cols))
ct_mlp.fit(X_train, y_train)

# %%
ct_mlp.model.eval()

# Use the SAME scaling your model uses
X_scaled = ct_mlp.scaler.transform(X_train)

# Use more samples for better comparison
X_bg = shap.kmeans(X_scaled, 100).data  # Extract data from DenseData object
X_sample = shap.utils.sample(X_scaled, 1000, 42)

# Convert to torch tensors
bg_t = torch.tensor(X_bg, dtype=torch.float32)
sample_t = torch.tensor(X_sample, dtype=torch.float32)

# GradientExplainer - more accurate than DeepExplainer
explainer = shap.GradientExplainer(ct_mlp.model, bg_t)
shap_values = explainer.shap_values(sample_t)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Ensure shap_values is 2D: (n_samples, n_features)
shap_values = np.array(shap_values).squeeze()
print(f"Final shap_values shape: {shap_values.shape}")

# Summary plot
shap.summary_plot(shap_values, features=X_sample, feature_names=feature_cols)

# %%
