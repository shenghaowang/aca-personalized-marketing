import random

import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklift.metrics import qini_auc_score
from tqdm import tqdm

from data.starbucks import load_data
from experiment.aca import experiment
from metrics.ranking import plot_uplift_curve
from model.feature_analysis import compute_shap_values, report_feature_contribution
from model.model_type import ModelType, get_model_kwargs, init_model


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # load data
    train_df, test_df = load_data(
        "data/starbucks/training.csv", "data/starbucks/test.csv"
    )

    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")

    feature_cols = [col for col in train_df.columns if col.startswith("V")]

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    logger.info("Training uplift model...")

    # Get model kwargs based on model type
    model_kwargs = get_model_kwargs(cfg.model, feature_cols)

    # Initialize and fit the model
    model = init_model(cfg.model, **model_kwargs)

    match cfg.model:
        case ModelType.ClassTransformLGBM.value | ModelType.ClassTransformMLP.value:
            y_train = train_df["label"].values
            model.fit(X_train, y_train)
            # predict uplift
            test_df["uplift"] = 2 * model.predict_proba(X_test)[:, 1] - 1
        case ModelType.UpliftRF.value:
            t_train = train_df["Promotion"].values
            y_train = train_df["purchase"].values
            model.fit(X=X_train, treatment=t_train, y=y_train)
            test_df["uplift"] = model.predict(X_test)
        case _:
            raise ValueError(f"Unsupported model type: {cfg.model}")

    # Analyze feature importance
    logger.info("Computing SHAP values for feature importance...")
    shap_vals, X_used = compute_shap_values(model, X_train)
    feature_impact_df = report_feature_contribution(shap_vals, X_used, feature_cols)
    logger.info(f"Feature impact:\n{feature_impact_df}")

    auqc = qini_auc_score(test_df["purchase"], test_df["uplift"], test_df["treatment"])
    logger.info(f"Qini coefficient on test data: {auqc:.4f}")

    plot_uplift_curve(test_df["purchase"], test_df["uplift"], test_df["treatment"])

    test_df["rank"] = (
        test_df["uplift"].rank(method="dense", ascending=False).astype(int)
    )
    test_df["normalised_rank"] = test_df["rank"] / test_df["rank"].max()

    results_list = []
    n_experiments = cfg.experiment.n_experiments
    for i in tqdm(range(n_experiments)):
        logger.info(f"Experiment {i+1}/{n_experiments}")
        res = experiment(
            model=init_model(cfg.model, **model_kwargs),
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            attack_attr=cfg.experiment.attack_attr,
            attack_val=cfg.experiment.attack_val,
            new_val=cfg.experiment.new_val,
            frac=cfg.experiment.frac,
        )
        logger.debug(res)
        results_list.append(res)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(cfg.experiment.results_dir, index=False)


if __name__ == "__main__":
    main()
