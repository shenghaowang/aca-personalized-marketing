import hydra
import torch
from loguru import logger
from omegaconf import DictConfig

from census_data import Census1990DataModule

MAX_EPOCHS = 1
N_ITER = 3


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    processed_data_dir = cfg.datasets.processed
    target_cols = [cfg.features.hour_col, cfg.features.gain_col, cfg.features.cost_col]

    # Test data loaders
    device = torch.device("cpu")
    data_module = Census1990DataModule(
        batch_size=32,
        train_dir=processed_data_dir.train,
        valid_dir=processed_data_dir.valid,
        test_dir=processed_data_dir.test,
        target_cols=target_cols,
    )

    training_iterator = iter(data_module.train_dataloader())
    validation_iterator = iter(data_module.val_dataloader())
    for epoch in range(MAX_EPOCHS):
        logger.info("\n")
        logger.info(f"==================== Epoch {epoch} ====================")
        # Training
        for _ in range(N_ITER):
            local_batch = next(training_iterator)
            local_features = local_batch["features"].to(device)
            local_treatment = local_batch["treatment"].to(device)
            local_gain = local_batch["gain"].to(device)
            local_cost = local_batch["cost"].to(device)
            logger.info(f"Batch features for training: {local_features.size()}")
            logger.info(f"Batch treatment for training: {local_treatment.size()}")
            logger.info(f"Batch gain for training: {local_gain.size()}")
            logger.info(f"Batch cost for training: {local_cost.size()}")

        # Validation
        with torch.set_grad_enabled(False):
            for _ in range(N_ITER):
                local_batch = next(validation_iterator)
                local_features = local_batch["features"].to(device)
                local_treatment = local_batch["treatment"].to(device)
                local_gain = local_batch["gain"].to(device)
                local_cost = local_batch["cost"].to(device)
                logger.info(f"Batch features for validation: {local_features.size()}")
                logger.info(f"Batch treatment for validation: {local_treatment.size()}")
                logger.info(f"Batch gain for validation: {local_gain.size()}")
                logger.info(f"Batch cost for validation: {local_cost.size()}")


if __name__ == "__main__":
    main()
