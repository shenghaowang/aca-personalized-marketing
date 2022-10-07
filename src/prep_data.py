from typing import List

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.debug(OmegaConf.to_container(cfg))
    df = pd.read_csv(cfg.datasets.raw)
    logger.info(f"Loaded data: {df.shape}")
    logger.info(f"\n{df.head()}")

    # Screening features
    filtered_df = select_features(df, cfg.features)

    # Create treatment variable
    treatment_col = create_treatment(working_hrs=filtered_df[cfg.features.hour_col])
    filtered_df[cfg.features.hour_col] = treatment_col
    logger.info(f"\n{filtered_df.head()}")
    logger.info(
        "Check the distribution of the treatment:\n"
        + f"{filtered_df[cfg.features.hour_col].value_counts()}"
    )

    # Create target variable for cost measurement
    filtered_df[cfg.features.cost_col] = filtered_df[cfg.features.cost_col] * (-1)
    logger.info(
        "Check the distribution of the cost:\n"
        + f"{filtered_df[cfg.features.cost_col].value_counts()}"
    )

    # Encode the categorical features
    encoded_df = encode(filtered_df, cfg.features.categorical_cols)

    # Split data for training, validation, and test
    split(encoded_df, cfg.features.hour_col, cfg.datasets.processed)


def select_features(df: pd.DataFrame, feature_cfg: DictConfig) -> pd.DataFrame:
    """Select the useful features and filter the data
    points with predefined eligibility criteria

    Parameters
    ----------
    df : pd.DataFrame
        raw US census data
    feature_cfg : DictConfig
        config params for feature screening

    Returns
    -------
    pd.DataFrame
        filtered data after removing the redundant
        features and rows
    """
    required_cols = (
        [feature_cfg.hour_col, feature_cfg.gain_col, feature_cfg.cost_col]
        + feature_cfg.binary_cols
        + feature_cfg.categorical_cols
    )
    logger.info(f"{len(required_cols)} fields will be taken from the raw data.")

    filtered_df = df[
        (df[feature_cfg.age_col] < 5)
        & (df[feature_cfg.citizen_col] == 0)
        & (df[feature_cfg.cost_col] >= 2)
    ][required_cols]
    logger.info(f"Processed data: {filtered_df.shape}")
    logger.info(f"\n{filtered_df.head()}")

    return filtered_df


def create_treatment(working_hrs: pd.Series) -> List[int]:
    """Create a treatment variable based on the
    working hours

    Parameters
    ----------
    working_hrs : pd.Series
        working_hrs feature

    Returns
    -------
    List[int]
        binary treatment variable. 1 indicates treated.
    """
    median_hrs = working_hrs.median()
    logger.info(f"median working hours = {median_hrs}")
    treatment = working_hrs.apply(lambda x: 1 if x > median_hrs else 0).tolist()
    logger.debug(treatment[:10])

    return treatment


def encode(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """One hot encode the categorical features

    Parameters
    ----------
    df : pd.DataFrame
        dataset to be processed

    Returns
    -------
    pd.DataFrame
        output dataset with the one-hot encoded features
    """
    df_no_encode = df[df.columns.difference(categorical_cols, sort=False)]
    logger.debug(f"df_no_encode: {df_no_encode.shape}")
    df_encoded = pd.get_dummies(df[categorical_cols], columns=categorical_cols)
    logger.debug(f"df_encoded: {df_encoded.shape}")

    combined_df = pd.concat([df_no_encode, df_encoded], axis=1)
    logger.info(f"One hot encoded data: {combined_df.shape}")
    logger.info(f"\n{combined_df.head()}")

    # Quality check the encoded data
    info_df = pd.DataFrame(combined_df.dtypes)
    info_df["MissingVal"] = combined_df.isnull().sum()
    info_df["NUnique"] = combined_df.nunique()
    info_df["Count"] = combined_df.count()
    info_df = info_df.rename(columns={0: "DataType"})

    pd.set_option("display.max_rows", None)
    logger.info(f"Screen the columns after encoding:\n{info_df}")

    return combined_df


def split(df: pd.DataFrame, stratify_col: str, processed_data_dir: DictConfig) -> None:
    """Split data for training, validation and test

    Parameters
    ----------
    df : pd.DataFrame
        data to split
    stratify_col : str
        treamtment column to be stratified against
    processed_data_dir : DictConfig
        output path of the processed data
    """
    train_df, rest_df = train_test_split(
        df,
        test_size=0.4,
        stratify=df[stratify_col],
        random_state=42,
    )
    valid_df, test_df = train_test_split(
        rest_df,
        test_size=0.5,
        stratify=rest_df[stratify_col],
        random_state=42,
    )
    logger.info(f"Training data: {train_df.shape}")
    logger.info(f"Validation data: {valid_df.shape}")
    logger.info(f"Test data: {test_df.shape}")

    # Export data
    train_df.to_csv(processed_data_dir.train, index=False)
    valid_df.to_csv(processed_data_dir.valid, index=False)
    test_df.to_csv(processed_data_dir.test, index=False)


if __name__ == "__main__":
    main()
