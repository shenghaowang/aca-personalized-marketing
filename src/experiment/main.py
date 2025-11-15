import importlib

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklift.metrics import qini_auc_score
from tqdm import tqdm

from experiment.aca import experiment
from model.trainer import load_model, predict_uplift

# Register a custom resolver to extract feature name from collective criterion
OmegaConf.register_new_resolver(
    "get_collective_feature",
    lambda x: list(x[0].keys())[0] if x and len(x) > 0 else "unknown",
)

# Register a custom resolver to format frac as percentage
OmegaConf.register_new_resolver("frac_to_pct", lambda x: f"{int(x * 100)}")


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

    # Run ACA experiments with the loaded model
    logger.info(f"Starting {cfg.experiment.n_experiments} ACA experiments...")
    results_list = []
    n_experiments = cfg.experiment.n_experiments

    # Get collective criterion and attack mappings from config
    collective_criterion = cfg.experiment.collective[0]  # Single eligibility criterion
    attack_mappings = cfg.experiment.attack

    for i in tqdm(range(n_experiments)):
        logger.info(f"Experiment {i+1}/{n_experiments}")
        res = experiment(
            model=model,
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            model_name=cfg.model.name,
            data_cfg=cfg.data,
            collective_criterion=collective_criterion,
            attack_mappings=attack_mappings,
            frac=cfg.experiment.frac,
        )
        logger.debug(res)
        results_list.append(res)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(cfg.experiment.results_dir, index=False)
    logger.info(f"Results saved to {cfg.experiment.results_dir}")


if __name__ == "__main__":
    main()
