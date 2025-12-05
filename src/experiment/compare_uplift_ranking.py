import importlib
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklift.metrics import qini_auc_score

from eval.ranking import number_responses, uplift_curve
from experiment.aca import CollectiveAction, ExperimentInput, rank_users_after_action
from model.model_type import init_model
from model.trainer import load_model, predict_uplift
from utils.config_utils import register_custom_resolvers
from utils.plot_utils import model_labels

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

    # Compute Qini score
    qini_score_before = qini_auc_score(
        test_df[cfg.data.target_col], test_df["uplift"], test_df["treatment"]
    )
    logger.info(
        f"{model_labels[cfg.model.name]} Qini coefficient before collective action: {qini_score_before:.4f}"
    )

    # Add ranking columns
    test_df["rank"] = (
        test_df["uplift"].rank(method="dense", ascending=False).astype(int)
    )
    test_df["normalised_rank"] = test_df["rank"] / test_df["rank"].max()

    # Initialize a fresh model for each experiment
    model = init_model(
        model_cfg=cfg.model,
        input_dim=len(feature_cols),
        control_name=(
            cfg.data.control_name if hasattr(cfg.data, "control_name") else None
        ),
    )

    # Initialize collective action
    action = CollectiveAction(
        collective_criterion=cfg.experiment.collective[
            0
        ],  # Single eligibility criterion
        attacks=cfg.experiment.attack,
        frac=cfg.experiment.frac,
    )
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

    normalised_rank_df, qini_score_after = rank_users_after_action(experiment_input)
    logger.info(
        f"{model_labels[cfg.model.name]} Qini coefficient after collective action: {qini_score_after:.4f}"
    )

    logger.debug(f"Normalised rank dataframe:\n{normalised_rank_df.head()}")

    # Plot Qini curves before and after collective action
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    xs, ys = uplift_curve(
        test_df[cfg.data.target_col],
        normalised_rank_df["uplift"],
        test_df["treatment"],
        n_nodes=None,
    )
    ax.plot(
        xs,
        ys,
        label=f"Before collective action: {qini_score_before:.4f}",
        color="blue",
    )

    xs, ys = uplift_curve(
        test_df[cfg.data.target_col],
        normalised_rank_df["uplift_modified"],
        test_df["treatment"],
        n_nodes=None,
    )
    ax.plot(
        xs,
        ys,
        label=f"After collective action: {qini_score_after:.4f}",
        color="orange",
    )

    # Add random model baseline
    responses_target, rescaled_responses_control = number_responses(
        test_df[cfg.data.target_col], test_df["treatment"]
    )
    incr_responses = responses_target - rescaled_responses_control
    ax.plot(
        [0, len(test_df)],
        [0, incr_responses],
        label="Random",
        color="green",
        linestyle="--",
    )

    # Formatting
    ax.set_title(
        f"Qini Curves of {model_labels[cfg.model.name]} for {cfg.data.name} Dataset"
    )
    ax.set_xlabel("Number of individuals targeted")
    ax.set_ylabel("Cumulative uplift")
    ax.legend()
    ax.grid(True)

    # Legend with white background
    legend = ax.legend()
    legend.get_frame().set_facecolor("white")  # white background
    legend.get_frame().set_edgecolor("black")  # optional border
    legend.get_frame().set_alpha(1.0)  # fully opaque

    # Save figure
    plt.tight_layout()
    output_path = (
        Path(cfg.artifacts.dir)
        / f"{cfg.data.name}_{cfg.model.name}_ranking_comparison.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    logger.info(f"Ranking comparison saved to {output_path}")


if __name__ == "__main__":
    main()
