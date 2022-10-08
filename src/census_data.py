from typing import List

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class Census1990Dataset(Dataset):
    def __init__(self, data_dir: str, target_cols: List[str]):
        """Initialize dataset for training, validation, test

        Parameters
        ----------
        data_dir : str
            directory of the processed data
        target_cols : List[str]
            treatment, gain, and cost columns
        """
        df = pd.read_csv(data_dir)
        hour_col, gain_col, cost_col = target_cols
        feature_cols = [col for col in df.columns if col not in target_cols]

        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.t = torch.tensor(df[hour_col].values, dtype=torch.float32)
        self.g = torch.tensor(df[gain_col].values, dtype=torch.float32)
        self.c = torch.tensor(df[cost_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return {
            "features": self.X[idx],
            "treatment": self.t[idx],
            "gain": self.g[idx],
            "cost": self.c[idx],
        }


class Census1990DataModule(pl.LightningDataModule):
    """LightningDataModule: Wrapper class for the dataset to be used in training"""

    def __init__(
        self,
        batch_size,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        target_cols: List[str],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.census_train = Census1990Dataset(train_dir, target_cols)
        self.census_valid = Census1990Dataset(valid_dir, target_cols)
        self.census_test = Census1990Dataset(test_dir, target_cols)

    def collate_fn(self, batch):
        """Convert the input raw data from the dataset into model input"""
        features, treatment, gain, cost = zip(
            *[
                (item["features"], item["treatment"], item["gain"], item["cost"])
                for item in batch
            ]
        )

        return {
            "features": torch.stack(features),
            "treatment": torch.stack(treatment),
            "gain": torch.stack(gain),
            "cost": torch.stack(cost),
        }

    def train_dataloader(self):
        return DataLoader(
            self.census_train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.census_valid,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.census_test,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
