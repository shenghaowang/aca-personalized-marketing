import importlib
import pickle
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklift.metrics import qini_auc_score

from eval.ranking import number_responses, uplift_curve
from model.trainer import predict_uplift
from utils.plot_utils import colors, model_labels

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
torch.set_num_threads(1)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="plot_config",
)
def main(cfg: DictConfig):
    """Plot Qini curves for different uplift models."""

    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # Load data
    data_loader = importlib.import_module("data.data_loader")
    _, test_df, _ = getattr(data_loader, cfg.data.load_data_method)(cfg.data)
    logger.info(f"Test data shape: {test_df.shape}")

    # Define model types to plot
    model_types = ["lgbm", "mlp", "uplift_rf"]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Get artifacts directory
    artifacts_dir = Path(cfg.artifacts.dir)

    # Plot Qini curve for each model
    for model_type in model_types:
        # Construct model artifact path
        model_file = f"{cfg.data.name}_{model_type}_model.pkl"
        features_file = f"{cfg.data.name}_{model_type}_features.pkl"

        model_path = artifacts_dir / model_file
        features_path = artifacts_dir / features_file

        # Check if model exists
        if not model_path.exists() or not features_path.exists():
            logger.warning(f"Model artifacts not found for {model_type}, skipping...")
            continue

        # Load pretrained model and features
        logger.info(f"Loading {model_type} model from {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(features_path, "rb") as f:
            feature_cols = pickle.load(f)

        # Predict uplift using the trainer utility
        test_df_with_uplift = predict_uplift(
            model=model,
            test_df=test_df.copy(),
            feature_cols=feature_cols,
            model_name=model_type,
        )
        uplift_pred = test_df_with_uplift["uplift"].values

        # Calculate Qini score
        qini_score = qini_auc_score(
            test_df[cfg.data.target_col], uplift_pred, test_df["treatment"]
        )
        logger.info(f"{model_labels[model_type]} Qini coefficient: {qini_score:.4f}")

        # Plot Qini curve
        xs, ys = uplift_curve(
            test_df[cfg.data.target_col],
            uplift_pred,
            test_df["treatment"],
            n_nodes=None,
        )
        ax.plot(
            xs,
            ys,
            label=f"{model_labels[model_type]}: {qini_score:.4f}",
            color=colors[model_type],
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
    ax.set_title(f"Qini Curves for {cfg.data.name} Dataset")
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
    output_path = artifacts_dir / f"{cfg.data.name}_qini_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    logger.info(f"Qini curves saved to {output_path}")


if __name__ == "__main__":
    main()
