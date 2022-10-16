import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig

from census_data import Census1990DataModule
from drm import DirectRankingModel, UserTargetingModel


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    processed_data_dir = cfg.datasets.processed
    hyparams = cfg.model
    target_cols = [cfg.features.hour_col, cfg.features.gain_col, cfg.features.cost_col]

    # Load fitted DRM model
    model = UserTargetingModel(
        model=DirectRankingModel(hyparams.input_dim, hyparams.hidden_dim),
        hyparams=hyparams,
    )
    model.load_state_dict(torch.load(cfg.model_file))
    model.eval()

    # Make test predictions
    data_module = Census1990DataModule(
        batch_size=hyparams.batch_size,
        train_dir=processed_data_dir.train,
        valid_dir=processed_data_dir.valid,
        test_dir=processed_data_dir.test,
        target_cols=target_cols,
    )
    trainer = pl.Trainer(max_epochs=hyparams.max_epochs, check_val_every_n_epoch=1)
    output = trainer.predict(model, data_module.test_dataloader())

    test_treatments = torch.cat([pred["treatment"] for pred in output])
    test_preds = torch.cat([pred["scores"] for pred in output])
    test_gain = torch.cat([pred["gain"] for pred in output])
    test_cost = torch.cat([pred["cost"] for pred in output])

    res = pd.DataFrame(
        data={
            "treatment": test_treatments.numpy(),
            "gain": test_gain.numpy(),
            "cost": test_cost.numpy(),
            "score": test_preds.numpy(),
        }
    )
    logger.info(f"Test predictions: {res.shape}")
    logger.info(f"\n{res.head()}")


if __name__ == "__main__":
    main()
