import hydra
import torch
from loguru import logger
from omegaconf import DictConfig

from census_data import Census1990DataModule
from drm import DirectRankingModel, Scoring, UserTargetingModel


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

    # Load a sample batch
    training_iterator = iter(data_module.train_dataloader())
    local_batch = next(training_iterator)
    local_features = local_batch["features"].to(device)
    local_treatment = local_batch["treatment"].to(torch.int64).to(device)
    local_gain = local_batch["gain"].to(device)
    local_cost = local_batch["cost"].to(device)
    logger.debug(f"Batch features for training: {local_features.size()}")
    logger.debug(f"Batch treatment for training: {local_treatment.size()}")
    logger.debug(f"Batch gain for training: {local_gain.size()}")
    logger.debug(f"Batch cost for training: {local_cost.size()}")

    # Test the direct ranking architecture
    drm = DirectRankingModel(input_dim=215, hidden_dim=64)
    res = drm(x=local_features, T=local_treatment)
    logger.info(f"Model output: {res.size()}")

    # Test the loss function
    utm = UserTargetingModel(model=drm)
    utm.calculate_loss(local_gain, local_cost, res, local_treatment)


def test_scoring() -> None:
    scoring = Scoring()
    s = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    T = torch.tensor([1, 1, 0, 0, 1])
    res = scoring(s, T)

    logger.debug(res)


if __name__ == "__main__":
    test_scoring()
    main()
