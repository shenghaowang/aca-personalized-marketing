import os

import numpy as np
import pandas as pd
from sklift.datasets import fetch_hillstrom

work_dir = os.path.dirname(os.path.dirname(__file__))
raw_data = fetch_hillstrom(target_col="all")
print(raw_data)
print("\n")
for key, val in raw_data.items():
    print(key)

print(f"Feature names: {raw_data['feature_names']}")
print(f"Target names: {raw_data['target_name']}")
print(f"Treatment name: {raw_data['treatment_name']}")

df = pd.DataFrame(
    data=np.c_[raw_data["data"], raw_data["treatment"], raw_data["target"]],
    columns=raw_data["feature_names"]
    + [raw_data["treatment_name"]]
    + raw_data["target_name"],
)
print(df.shape)
print(df.head())

df.to_csv("data/hillstrom-email-analytics.csv", index=False)
