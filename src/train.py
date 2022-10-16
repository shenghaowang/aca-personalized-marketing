from typing import List

import hydra
import pytorch_lightning as pl
import torch

# from loguru import logger
from omegaconf import DictConfig

from census_data import Census1990DataModule
from drm import DirectRankingModel, UserTargetingModel


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):

    # Initilize the direct ranking model
    drm = DirectRankingModel(
        input_dim=cfg.model.input_dim, hidden_dim=cfg.model.hidden_dim
    )
    target_cols = [cfg.features.hour_col, cfg.features.gain_col, cfg.features.cost_col]
    trainer(
        model=drm,
        processed_data_dir=cfg.datasets.processed,
        hyparams=cfg.model,
        target_cols=target_cols,
    )


def trainer(
    model: DirectRankingModel,
    processed_data_dir: DictConfig,
    hyparams: DictConfig,
    target_cols: List[str],
):
    # device = torch.device("cpu")

    # Create a pytorch trainer
    trainer = pl.Trainer(max_epochs=hyparams.max_epochs, check_val_every_n_epoch=1)

    torch.manual_seed(seed=42)

    data_module = Census1990DataModule(
        batch_size=32,
        train_dir=processed_data_dir.train,
        valid_dir=processed_data_dir.valid,
        test_dir=processed_data_dir.test,
        target_cols=target_cols,
    )

    # Instantiate a new model
    model = UserTargetingModel(model, hyparams)

    # Train and validate the model
    trainer.fit(
        model,
        data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    # Predict on the same test set to show some output
    trainer.predict(model, data_module.test_dataloader())


if __name__ == "__main__":
    main()
