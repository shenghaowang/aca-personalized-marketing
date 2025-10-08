import random

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklift.metrics import qini_auc_score
from tqdm import tqdm

from data.starbucks import load_data
from experiment.aca import experiment
from metrics.ranking import plot_uplift_curve
from model.neuralnet import NeuralNetClassifier


def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # load data
    train_df, test_df = load_data(
        "data/starbucks/training.csv", "data/starbucks/test.csv"
    )

    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")

    feature_cols = [col for col in train_df.columns if col.startswith("V")]

    # Train the original class transformation model
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    # t_train = train_df["Promotion"].values

    X_test = test_df[feature_cols].values

    # define approach
    logger.info("Training Class Transformation model...")
    ct = NeuralNetClassifier(input_dim=len(feature_cols))

    # fit the model
    ct.fit(X_train, y_train)

    # predict uplift
    test_df["uplift"] = 2 * ct.predict_proba(X_test)[:, 1] - 1

    auqc = qini_auc_score(test_df["purchase"], test_df["uplift"], test_df["Promotion"])
    logger.info(f"Qini coefficient on test data: {auqc:.4f}")

    plot_uplift_curve(test_df["purchase"], test_df["uplift"], test_df["Promotion"])

    test_df["rank"] = (
        test_df["uplift"].rank(method="dense", ascending=False).astype(int)
    )
    test_df["normalised_rank"] = test_df["rank"] / test_df["rank"].max()

    # res = experiment(
    #     train_df=train_df,
    #     test_df=test_df,
    #     feature_cols=feature_cols,
    #     attack_attr='V6',
    #     attack_seg=4,
    #     new_val=1,
    #     frac=0.1
    # )
    # logger.info(res)

    num_experiments = 100
    results_list = []
    for i in tqdm(range(num_experiments)):
        logger.info(f"Experiment {i+1}/{num_experiments}")
        res = experiment(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            attack_attr="V6",
            attack_seg=4,
            new_val=1,
            frac=0.1,
        )
        results_list.append(res)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv("results_nn.csv", index=False)


if __name__ == "__main__":
    main()
