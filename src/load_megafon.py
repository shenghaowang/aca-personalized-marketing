import numpy as np
import pandas as pd
from loguru import logger
from sklift.datasets import fetch_megafon


def run():
    df = load_data()
    logger.info(f"Loaded data: {df.shape}")
    logger.info(f"\n{df.head()}")

    return df


def load_data() -> pd.DataFrame:
    raw_ds = fetch_megafon()
    logger.info(f"Dataset type: {type(raw_ds)}\n")
    logger.info(f"Dataset features shape: {raw_ds.data.shape}")
    logger.info(f"Dataset target shape: {raw_ds.target.shape}")
    logger.info(f"Dataset treatment shape: {raw_ds.treatment.shape}")

    df = pd.concat([raw_ds.data, raw_ds.treatment, raw_ds.target], axis=1)

    return df


def simulate_spending(mean: float, std: float, size: int) -> np.array:
    """Simulate the spending for the transactions
    from a specific user group

    Spending from any user group is assumed to
    follow a specific normal distribution.

    Parameters
    ----------
    mean : float
        mean value of the spending
    std : float
        standard deviation of the spending
    size : int
        volume of transactions

    Returns
    -------
    np.array
        simulated spending
    """
    spending = np.random.normal(mean, std, size)

    return spending.clip(min=0)


if __name__ == "__main__":
    run()
