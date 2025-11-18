import importlib
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklift.metrics import qini_auc_score

from experiment.aca import CollectiveAction, ExperimentInput, rank_users_after_action
from model.model_type import init_model
from model.trainer import load_model, predict_uplift
from utils.config_utils import register_custom_resolvers
from utils.plot_utils import compare_distributions

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

    # Initialize a fresh model
    model = init_model(
        model_cfg=cfg.model,
        input_dim=len(feature_cols),
        control_name=(
            cfg.data.control_name if hasattr(cfg.data, "control_name") else None
        ),
    )

    # Run collective action experiment
    experiment_input = ExperimentInput(
        untrained_model=model,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        model_name=cfg.model.name,
        data_cfg=cfg.data,
        action=action,
        seed=42,
    )
    normalised_rank_df, auqc_modified = rank_users_after_action(input=experiment_input)

    logger.info(f"Modified Qini coefficient: {auqc_modified:.4f}")
    logger.info(f"Qini coefficient change: {auqc_modified - auqc:.4f}")

    # Compare distributions for all users
    compare_distributions(
        before=normalised_rank_df["normalised_rank"].values,
        after=normalised_rank_df["normalised_rank_modified"].values,
        output_path=Path("artifacts/rank_distribution_all_users.png"),
        xlabel="Normalised Rank",
        title="Normalised Rank Distribution Before and After Collective Action",
    )
    compare_distributions(
        before=normalised_rank_df["uplift"].values,
        after=normalised_rank_df["uplift_modified"].values,
        output_path=Path("artifacts/uplift_distribution_all_users.png"),
        xlabel="Uplift Score",
        title="Uplift Score Distribution Before and After Collective Action",
    )

    # Compare distributions for participants only
    collective_df = normalised_rank_df[normalised_rank_df["aca_flag"] == 1]
    logger.info(f"Number of participants: {collective_df.shape[0]}")

    compare_distributions(
        before=collective_df["normalised_rank"].values,
        after=collective_df["normalised_rank_modified"].values,
        output_path=Path("artifacts/rank_distribution_participants.png"),
        xlabel="Normalised Rank",
        title="Normalised Rank Distribution of Collective Action Participants",
    )
    compare_distributions(
        before=collective_df["normalised_rank"].values,
        after=collective_df["normalised_rank_modified"].values,
        output_path=Path("artifacts/rank_distribution_participants.png"),
        xlabel="Normalised Rank",
        title="Normalised Rank Distribution of Collective Action Participants",
    )

    compare_distributions(
        before=collective_df["uplift"].values,
        after=collective_df["uplift_modified"].values,
        output_path=Path("artifacts/uplift_distribution_participants.png"),
        xlabel="Uplift Score",
        title="Uplift Score Distribution of Collective Action Participants",
    )


if __name__ == "__main__":
    main()
