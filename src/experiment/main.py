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
from model.model_type import init_model


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

    # Train the original class transformation model
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    # t_train = train_df["Promotion"].values

    X_test = test_df[feature_cols].values

    # define approach
    logger.info("Training Class Transformation model...")
    ct = init_model(cfg.model, input_dim=len(feature_cols))

    # fit the model
    ct.fit(X_train, y_train)

    # predict uplift
    test_df["uplift"] = 2 * ct.predict_proba(X_test)[:, 1] - 1

    auqc = qini_auc_score(test_df["purchase"], test_df["uplift"], test_df["Promotion"])
    logger.info(f"Qini coefficient on test data: {auqc:.4f}")

    plot_uplift_curve(test_df["purchase"], test_df["uplift"], test_df["Promotion"])

    test_df["rank"] = (
        test_df["uplift"].rank(method="dense", ascending=False).astype(int)
    )
    test_df["normalised_rank"] = test_df["rank"] / test_df["rank"].max()

    results_list = []
    n_experiments = cfg.experiment.n_experiments
    for i in tqdm(range(n_experiments)):
        logger.info(f"Experiment {i+1}/{n_experiments}")
        res = experiment(
            model=init_model(cfg.model, input_dim=len(feature_cols)),
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
