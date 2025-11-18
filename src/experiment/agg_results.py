from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from utils.plot_utils import plot_pct_decrease


@hydra.main(version_base=None, config_path="../config", config_name="results_config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # Load and aggregate results
    csv_files = list(Path(cfg.results_dir).glob(cfg.file_path))

    if not csv_files:
        raise FileNotFoundError(
            f"No files found in {cfg.results_dir} matching {cfg.file_path}"
        )

    # Load and combine all CSV files
    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Loaded {len(csv_files)} files with {len(combined_df)} total rows")
    logger.debug(f"Combined DataFrame:\n{combined_df.tail()}")

    target_features = combined_df["target_feature"].unique()
    results = []
    for feature in target_features:
        df = combined_df[combined_df["target_feature"] == feature]
        avail_models = df["model"].unique().tolist()

        for model in avail_models:
            model_df = df[df["model"] == model]

            for frac in np.array(range(1, 11)) / 100:
                frac_df = model_df[model_df["frac"] == frac].copy()
                if frac_df.empty:
                    logger.warning(
                        f"No data for feature={feature}, model={model}, frac={frac}"
                    )
                    continue

                res = process_results(frac_df)
                res["dataset"] = frac_df["dataset"].iloc[0]
                res["model"] = model
                res["target_feature"] = feature
                res["frac"] = frac

                results.append(res)

    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(cfg.agg_results_file), index=False)
    logger.info(f"Aggregated results saved to {cfg.agg_results_file}")

    # Plot the percentage decrease in metrics
    for feature in target_features:
        feature_results = results_df[results_df["target_feature"] == feature]
        avail_models = feature_results["model"].unique().tolist()

        avg_delta_qini, ci_low, ci_high = [], [], []
        for model in avail_models:
            model_results = feature_results[
                feature_results["model"] == model
            ].sort_values("frac")
            avg_delta_qini.append(
                np.insert(model_results["avg_delta_qini"].values, 0, 0)
            )
            ci_low.append(np.insert(model_results["avg_ci_low"].values, 0, 0))
            ci_high.append(np.insert(model_results["avg_ci_high"].values, 0, 0))

        # Plot average Qini coefficient decrease
        plot_pct_decrease(
            model_names=avail_models,
            title=f"Average Qini Coefficient Decrease - Feature: {feature}",
            output_path=Path(cfg.artifacts_dir)
            / cfg.qini_coeff_chart.format(feature=feature),
            avg=avg_delta_qini,
        )

        # Derive average normalised rank decrease
        avg_norm_rank_decrease = []
        for i in range(len(avail_models)):
            avg_norm_rank_decrease.append((ci_low[i] + ci_high[i]) / 2)

        plot_pct_decrease(
            model_names=avail_models,
            title=f"Average Normalised Rank Decrease - Feature: {feature}",
            output_path=Path(cfg.artifacts_dir)
            / cfg.avg_norm_rank_chart.format(feature=feature),
            avg=avg_norm_rank_decrease,
            ci_low=ci_low,
            ci_high=ci_high,
        )


def process_results(df: pd.DataFrame):
    df["delta_qini"] = df.apply(
        lambda row: (row["baseline_qini_coeff"] - row["qini_coeff"])
        / row["baseline_qini_coeff"],
        axis=1,
    )

    res = {}
    for metric in ("ci_low", "ci_high", "delta_qini"):
        res[f"avg_{metric}"] = df[metric].mean()
        res[f"std_{metric}"] = df[metric].std()
        res[f"median_{metric}"] = df[metric].median()
        res[f"min_{metric}"] = df[metric].min()
        res[f"max_{metric}"] = df[metric].max()

    return res


if __name__ == "__main__":
    main()
