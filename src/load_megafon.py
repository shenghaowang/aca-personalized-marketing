from typing import Dict, Tuple

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklift.datasets import fetch_megafon


class MegafonDataReader:
    def __init__(
        self,
        treatment_col: str,
        conversion_col: str,
        spending_col: str,
        spending_params: Dict[str, Dict[str, float]],
        output_dir: DictConfig,
    ) -> None:
        """Process the raw Megafon data and split the sampled data
        for the subsequent model training

        Parameters
        ----------
        treatment_col : str
            name of the treatment group column
        conversion_col : str
            name of the conversion column
        spending_col : str
            name of the spending column
        spending_params : Dict[str, Dict[str, float]]
            parameters for simulating the spending
        output_dir : DictConfig
            output directories for the processed data
        """
        self.treatment_col = treatment_col
        self.conversion_col = conversion_col
        self.spending_col = spending_col
        self.stratify_cols = [treatment_col, conversion_col]
        self.spending_params = spending_params
        self.output_dir = output_dir

    def get_stratified_sample(self, df: pd.DataFrame, frac=0.2) -> pd.DataFrame:
        """Take a stratified sample against the treatment and
        conversion columns

        Parameters
        ----------
        df : pd.DataFrame
            full dataset
        frac : float, optional
            percentage sample size, by default 0.2

        Returns
        -------
        pd.DataFrame
            sampled data
        """

        return (
            df.groupby(self.stratify_cols, group_keys=False)
            .apply(lambda x: x.sample(frac=frac, random_state=1))
            .reset_index(drop=True)
        )

    def split_data_by_conversion(
        self, df: pd.DataFrame, treatment_val: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data based on whether there is a conversion

        Parameters
        ----------
        df : pd.DataFrame
            sample dataset
        treatment_val : str
            name of the target treatment group

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            split datasets for w/o a conversion
        """
        treatment_df = df[df[self.treatment_col] == treatment_val]
        mask = df[self.conversion_col] == 1

        return (
            treatment_df.loc[mask].sample(frac=1, random_state=1),
            treatment_df.loc[~mask].sample(frac=1, random_state=1),
        )

    @staticmethod
    def simulate_spending(mean: float, std: float, size: int) -> np.array:
        """Simulate the spending for the transactions
        from a specific user group

        Spending from any user group is assumed to
        follow a specific normal distribution.

        Parameters
        ----------
        mean : float
            mean value of the spending
        std : float
            standard deviation of the spending
        size : int
            volume of transactions

        Returns
        -------
        np.array
            simulated spending
        """
        spending = np.random.normal(mean, std, size)

        return spending.clip(min=0)

    def run(self, df: pd.DataFrame) -> None:
        # Take a stratified sample from the full dataset
        sample_df = self.get_stratified_sample(df)
        logger.info(f"Sample data: {sample_df.shape}")

        sample_df_with_spending = pd.DataFrame()
        for treatment_group in ["treatment", "control"]:
            logger.info(
                f"Simulate spending for group: {treatment_group} "
                + f"with {self.spending_params[treatment_group]}"
            )

            # Split the data by conversion
            converted_df, unconverted_df = self.split_data_by_conversion(
                sample_df, treatment_group
            )

            # Simulate the spending for the conversion
            converted_df[self.spending_col] = self.simulate_spending(
                mean=self.spending_params[treatment_group]["mean"],
                std=self.spending_params[treatment_group]["std"],
                size=len(converted_df),
            )
            unconverted_df[self.spending_col] = 0

            sample_df_with_spending = pd.concat(
                [sample_df_with_spending, converted_df, unconverted_df]
            )

        logger.info(
            f"Sample data with simulated spending: {sample_df_with_spending.shape}"
        )
        avg_spending_per_treatment = sample_df_with_spending.groupby(
            self.treatment_col
        ).agg({self.spending_col: "mean"})
        logger.info(
            f"Sanity check on the simulated spending:\n{avg_spending_per_treatment}"
        )

        sample_df_with_spending[self.treatment_col] = sample_df_with_spending[
            self.treatment_col
        ].map({"control": 0, "treatment": 1})

        # Split data for training, validation, and test
        train_df, rest_df = train_test_split(
            sample_df_with_spending,
            test_size=0.3,
            stratify=sample_df_with_spending[self.stratify_cols],
            random_state=42,
        )
        valid_df, test_df = train_test_split(
            rest_df,
            test_size=0.5,
            stratify=rest_df[self.stratify_cols],
            random_state=42,
        )
        logger.info(f"Training data: {train_df.shape}")
        logger.info(f"Validation data: {valid_df.shape}")
        logger.info(f"Test data: {test_df.shape}")

        # Export data
        train_df.to_csv(self.output_dir.train_dir, index=False)
        valid_df.to_csv(self.output_dir.valid_dir, index=False)
        test_df.to_csv(self.output_dir.test_dir, index=False)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.debug(OmegaConf.to_container(cfg))
    df = load_data()
    logger.info(f"Loaded data: {df.shape}")
    logger.info(f"\n{df.head()}")

    data_reader = MegafonDataReader(
        treatment_col=cfg.features.treatment_col,
        conversion_col=cfg.features.conversion_col,
        spending_col=cfg.features.spending_col,
        spending_params=OmegaConf.to_container(cfg.features.spending_simulation),
        output_dir=cfg.datasets.path,
    )
    data_reader.run(df)


def load_data() -> pd.DataFrame:
    """Import Megafon data via the sklift package
    into a dataframe

    Returns
    -------
    pd.DataFrame
        full Megafon dataset
    """
    raw_ds = fetch_megafon()
    logger.info(f"Dataset type: {type(raw_ds)}\n")
    logger.info(f"Dataset features shape: {raw_ds.data.shape}")
    logger.info(f"Dataset target shape: {raw_ds.target.shape}")
    logger.info(f"Dataset treatment shape: {raw_ds.treatment.shape}")

    df = pd.concat([raw_ds.data, raw_ds.treatment, raw_ds.target], axis=1)

    return df


if __name__ == "__main__":
    main()
