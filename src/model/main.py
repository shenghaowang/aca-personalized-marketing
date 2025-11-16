import importlib

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklift.metrics import qini_auc_score

from eval.feature_analysis import compute_shap_values, report_feature_contribution
from eval.ranking import plot_uplift_curve
from model.model_type import init_model
from model.trainer import save_model, train_and_predict
from utils.seed_utils import set_seed


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Main training script for uplift models."""

    # Skip the experiment config
    cfg = {k: v for k, v in cfg.items() if k != "experiment"}
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # Set random seeds for reproducibility
    set_seed()

    # Load data
    data_loader = importlib.import_module("data.data_loader")
    train_df, test_df, feature_cols = getattr(data_loader, cfg.data.load_data_method)(
        cfg.data
    )
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")

    logger.info("Training uplift model...")

    # Initialize model with config
    model = init_model(
        model_cfg=cfg.model,
        input_dim=len(feature_cols),
        control_name=(
            cfg.data.control_name if hasattr(cfg.data, "control_name") else None
        ),
    )

    # Train model and predict uplift
    test_df = train_and_predict(
        model=model,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        model_name=cfg.model.name,
        data_cfg=cfg.data,
    )

    # Evaluate model
    auqc = qini_auc_score(
        test_df[cfg.data.target_col], test_df["uplift"], test_df["treatment"]
    )
    logger.info(f"Qini coefficient on test data: {auqc:.4f}")

    # Plot uplift curve
    plot_uplift_curve(
        test_df[cfg.data.target_col], test_df["uplift"], test_df["treatment"]
    )

    # Analyze feature importance
    logger.info("Computing SHAP values for feature importance...")
    X_train = train_df[feature_cols].values
    shap_vals, X_used, X_original = compute_shap_values(model, X_train)
    feature_impact_df = report_feature_contribution(
        shap_vals, X_used, feature_cols, X_original
    )
    logger.info(f"Feature impact:\n{feature_impact_df}")

    # Save trained model
    saved_paths = save_model(
        model=model,
        feature_cols=feature_cols,
        artifacts_cfg=cfg.artifacts,
    )
    logger.info(f"Training completed. Artifacts saved: {saved_paths}")


if __name__ == "__main__":
    main()
