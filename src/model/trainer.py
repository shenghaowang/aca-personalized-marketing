import pickle
from pathlib import Path
from typing import List, Union

import pandas as pd
from causalml.inference.tree import UpliftRandomForestClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from omegaconf import DictConfig

from model.model_type import ModelType
from model.neuralnet import NeuralNetClassifier


def train_and_predict(
    model: Union[LGBMClassifier, NeuralNetClassifier, UpliftRandomForestClassifier],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    data_cfg: DictConfig,
) -> pd.DataFrame:
    """Train a model and predict uplift scores on test data.

    Args:
        model: Model instance to train
        train_df: Training dataframe
        test_df: Test dataframe
        feature_cols: List of feature column names
        model_name: Model type name (lgbm, mlp, uplift_rf)
        data_cfg: Data configuration containing treatment_col and target_col

    Returns:
        Test dataframe with 'uplift' column added
    """
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    match model_name:
        case ModelType.ClassTransformLGBM.value | ModelType.ClassTransformMLP.value:
            # Class transformation models
            y_train = train_df["label"].values
            model.fit(X_train, y_train)
            test_df["uplift"] = 2 * model.predict_proba(X_test)[:, 1] - 1
        case ModelType.UpliftRF.value:
            # Uplift Random Forest
            t_train = train_df[data_cfg.treatment_col].values
            y_train = train_df[data_cfg.target_col].values
            model.fit(X=X_train, treatment=t_train, y=y_train)
            test_df["uplift"] = model.predict(X_test)
        case _:
            raise ValueError(f"Unsupported model type: {model_name}")

    return test_df


def predict_uplift(
    model: Union[LGBMClassifier, NeuralNetClassifier, UpliftRandomForestClassifier],
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
) -> pd.DataFrame:
    """Predict uplift scores using a pre-trained model.

    Args:
        model: Pre-trained model instance
        test_df: Test dataframe
        feature_cols: List of feature column names
        model_name: Model type name (lgbm, mlp, uplift_rf)

    Returns:
        Test dataframe with 'uplift' column added
    """
    X_test = test_df[feature_cols].values

    match model_name:
        case ModelType.ClassTransformLGBM.value | ModelType.ClassTransformMLP.value:
            # Class transformation models
            test_df["uplift"] = 2 * model.predict_proba(X_test)[:, 1] - 1
        case ModelType.UpliftRF.value:
            # Uplift Random Forest
            test_df["uplift"] = model.predict(X_test)
        case _:
            raise ValueError(f"Unsupported model type: {model_name}")

    return test_df


def save_model(
    model: Union[LGBMClassifier, NeuralNetClassifier, UpliftRandomForestClassifier],
    feature_cols: List[str],
    collective_actions_report: pd.DataFrame,
    artifacts_cfg: DictConfig,
) -> dict:
    """Save trained uplift model and associated metadata.

    Args:
        model: Trained model instance
        feature_cols: List of feature column names used for training
        artifacts_cfg: Artifacts configuration from Hydra config

    Returns:
        Dictionary containing paths to saved artifacts
    """
    artifacts_dir = Path(artifacts_cfg.dir)
    artifacts_dir.mkdir(exist_ok=True, parents=True)

    # Save model
    model_path = artifacts_dir / artifacts_cfg.model_file
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")

    # Save feature columns
    features_path = artifacts_dir / artifacts_cfg.features_file
    with open(features_path, "wb") as f:
        pickle.dump(feature_cols, f)
    logger.info(f"Feature columns saved to {features_path}")

    # Export collective actions report
    ca_report_path = artifacts_dir / artifacts_cfg.collective_actions
    collective_actions_report.to_csv(ca_report_path, index=False)

    return {
        "model_path": str(model_path),
        "features_path": str(features_path),
        "collective_actions_path": str(ca_report_path),
    }


def load_model(artifacts_cfg: DictConfig) -> tuple:
    """Load a saved uplift model and its feature columns.

    Args:
        artifacts_cfg: Artifacts configuration from Hydra config

    Returns:
        Tuple of (model, feature_cols)
    """
    artifacts_dir = Path(artifacts_cfg.dir)

    # Load model
    model_path = artifacts_dir / artifacts_cfg.model_file
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")

    # Load feature columns
    features_path = artifacts_dir / artifacts_cfg.features_file
    with open(features_path, "rb") as f:
        feature_cols = pickle.load(f)
    logger.info(f"Feature columns loaded from {features_path}")

    return model, feature_cols
