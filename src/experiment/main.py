import importlib

import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklift.metrics import qini_auc_score
from tqdm import tqdm

from experiment.aca import CollectiveAction, ExperimentInput, experiment
from model.model_type import init_model
from model.trainer import load_model, predict_uplift
from utils.config_utils import register_custom_resolvers
from utils.seed_utils import set_seed

torch.set_num_threads(1)

# Register custom resolvers for Hydra/OmegaConf
register_custom_resolvers()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # Load data
    data_loader = importlib.import_module("data.data_loader")
    train_df, test_df, _ = getattr(data_loader, cfg.data.load_data_method)(cfg.data)
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")

    # Load pre-trained model
    logger.info("Loading pre-trained model...")
    model, feature_cols = load_model(cfg.artifacts)
    logger.info(f"Successfully loaded model with {len(feature_cols)} features")

    # Run baseline prediction to get initial rankings
    logger.info("Running baseline predictions...")
    test_df = predict_uplift(
        model=model,
        test_df=test_df,
        feature_cols=feature_cols,
        model_name=cfg.model.name,
    )

    auqc = qini_auc_score(
        test_df[cfg.data.target_col], test_df["uplift"], test_df["treatment"]
    )
    logger.info(f"Baseline Qini coefficient: {auqc:.4f}")

    test_df["rank"] = (
        test_df["uplift"].rank(method="dense", ascending=False).astype(int)
    )
    test_df["normalised_rank"] = test_df["rank"] / test_df["rank"].max()

    # Get collective criterion and attack recipe from config
    action = CollectiveAction(
        collective_criterion=cfg.experiment.collective[
            0
        ],  # Single eligibility criterion
        attacks=cfg.experiment.attack,
        frac=cfg.experiment.frac,
    )

    # Generate random seeds for each experiment
    np.random.seed(42)
    n_experiments = cfg.experiment.n_experiments
    seeds = np.random.randint(0, 10000, size=n_experiments)

    # Run ACA experiments
    logger.info(f"Starting {n_experiments} ACA experiments...")
    results_list = []
    for i in tqdm(range(n_experiments)):
        logger.info(f"Experiment {i+1}/{n_experiments}")

        # Set random seeds for reproducibility
        set_seed()

        # Initialize a fresh model for each experiment
        model = init_model(
            model_cfg=cfg.model,
            input_dim=len(feature_cols),
            control_name=(
                cfg.data.control_name if hasattr(cfg.data, "control_name") else None
            ),
        )
        experiment_input = ExperimentInput(
            untrained_model=model,
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            model_name=cfg.model.name,
            data_cfg=cfg.data,
            action=action,
            seed=int(seeds[i]),
        )
        res = experiment(input=experiment_input)
        res["baseline_qini_coeff"] = auqc

        logger.debug(res)
        results_list.append(res)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(cfg.experiment.results_dir, index=False)
    logger.info(f"Results saved to {cfg.experiment.results_dir}")


if __name__ == "__main__":
    main()
