from os import listdir
from typing import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

from qamsi.config.base_experiment_config import BaseExperimentConfig
from qamsi.config.topn_experiment_config import TopNExperimentConfig
from qamsi.utils.data import read_csv
from qamsi.utils.corr import avg_corr


def _load_data(config: BaseExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    filename = config.PREFIX + config.DF_FILENAME
    pm_filename = config.PREFIX + config.PRESENCE_MATRIX_FILENAME
    data = read_csv(config.PATH_OUTPUT, filename)

    if not data.index.is_unique:
        msg = "Returns have non-unique dates!"
        raise ValueError(msg)

    presence_matrix = read_csv(config.PATH_OUTPUT, pm_filename)

    return data, presence_matrix


def _rolling_feature(
    df: pd.DataFrame,
    feature_fn: Callable,
    presense_matrix: pd.DataFrame,
    feature_name: str | None = None,
    verbose: bool = False,
):
    """Function to compute rolling correlation."""
    # Initialize a list to store results
    results = []

    # Perform calculation for each rolling window
    for end in tqdm(df.index) if verbose else df.index:
        start = end - pd.DateOffset(months=1)

        curr_matrix = presense_matrix.loc[:end].iloc[-1]
        selection = curr_matrix[curr_matrix == 1].index.tolist()
        rolling_window = df[selection].loc[start:end]

        feature = feature_fn(rolling_window)

        results.append([end, feature])

    # Create a series with the results
    feature_name = "feature" if feature_name is None else feature_name
    rolling_feat = pd.DataFrame(results, columns=["date", feature_name])
    rolling_feat["date"] = pd.to_datetime(rolling_feat["date"])
    rolling_feat = rolling_feat.set_index("date")

    return rolling_feat


def _compute_dnk_features(
    ret: pd.DataFrame,
    presence_matrix: pd.DataFrame,
    filename: Path,
    verbose: bool = False,
) -> pd.DataFrame:
    # 1. Avg Corr.
    # Calculate rolling average correlation of non-diagonal elements
    rolling_avg_corr = _rolling_feature(
        ret, avg_corr, presence_matrix, "avg_corr", verbose=verbose
    )

    # 2. Average volatility.
    avg_vol = _rolling_feature(
        ret, lambda s: s.std(axis=0).mean(), presence_matrix, "avg_vol", verbose=verbose
    )

    # 3. EW Portfolio.
    ew = _rolling_feature(
        ret,
        lambda s: np.prod(1 + np.nanmean(s, axis=1)) - 1,
        presence_matrix,
        "ew",
        verbose=verbose,
    )

    # 4. EW Portfolio Moving Average.
    ewma = []
    for end in tqdm(ew.index) if verbose else ew.index:
        start = end - pd.DateOffset(months=1)

        if end > ew.index[-1]:
            break

        sample = ew.loc[start:end]

        ma = sample.ewm(alpha=0.1).mean().iloc[-1].item()

        ewma.append([end, ma])
    ewma = pd.DataFrame(ewma, columns=["date", "ewma"])
    ewma["date"] = pd.to_datetime(ewma["date"])
    ewma = ewma.set_index("date")

    # 4. Ledoit-Wolf Shrinkage Intensity.
    def get_intensity(s: pd.DataFrame):
        s = s.copy().fillna(0)
        lw_estimator = LedoitWolf()
        lw_estimator.fit(s)
        return lw_estimator.shrinkage_

    lw = _rolling_feature(
        ret,
        lambda s: get_intensity(s),
        presence_matrix,
        "lw_shrinkage",
        verbose=verbose,
    )

    # 5. Momentum
    momentum = _rolling_feature(
        ret,
        lambda s: np.nanmean(np.where(s, s > 0, 1), axis=0).mean(),
        presence_matrix,
        "momentum_feature",
        verbose=verbose,
    )

    # 6. Trace.
    trace = _rolling_feature(
        ret,
        lambda s: np.trace(s.fillna(0).cov()),
        presence_matrix,
        "trace",
        verbose=verbose,
    )

    # 7. Universe Volatility.
    ew_vol = ew.rolling(window=252, min_periods=1).std().fillna(0)

    # Merge all features.
    features = rolling_avg_corr.merge(
        avg_vol, how="inner", left_index=True, right_index=True
    )

    features = features.merge(ewma, how="inner", left_index=True, right_index=True)
    features = features.merge(lw, how="inner", left_index=True, right_index=True)
    features = features.merge(momentum, how="inner", left_index=True, right_index=True)
    features = features.merge(trace, how="inner", left_index=True, right_index=True)
    features = features.merge(
        ew_vol.rename(columns={"ew": "universe_vol"}),
        how="inner",
        left_index=True,
        right_index=True,
    )

    if rolling_avg_corr.shape[0] != features.shape[0]:
        msg = "The dates of created features do not match!"
        raise ValueError(msg)

    features.to_csv(filename)

    return features


def create_dnk_features_targets(
    config: TopNExperimentConfig, verbose: bool = False
) -> None:
    data, presence_matrix = _load_data(config)
    ret = data[presence_matrix.columns]

    features_filename = config.PREFIX + config.DNK_FEATURES_TMP_FILENAME
    if features_filename not in listdir(config.PATH_TMP):
        _compute_dnk_features(
            ret,
            presence_matrix,
            filename=config.PATH_TMP / features_filename,
            verbose=verbose,
        )

    dnk_features = read_csv(config.PATH_TMP, features_filename)

    targets = pd.read_csv(config.PATH_TARGETS / f"targets_{config.TOPN}.csv")
    targets["start_date"] = pd.to_datetime(targets["start_date"])
    targets["end_date"] = pd.to_datetime(targets["end_date"])

    dnk_data = targets.merge(
        dnk_features, how="right", left_on="start_date", right_index=True
    )

    dnk_data = dnk_data.rename(columns={"start_date": "date"})
    dnk_data = dnk_data.set_index("date")

    dnk_data = dnk_data.rename(columns={"shrinkage": "target"})

    full_df = data.merge(dnk_data, left_index=True, right_index=True)
    full_df.to_csv(config.PATH_OUTPUT / (config.PREFIX + config.DF_FILENAME))


if __name__ == "__main__":
    from run import Dataset

    TOP_N = 30
    dataset = Dataset.TOPN_US

    settings = dataset.value(topn=TOP_N)
    create_dnk_features_targets(settings)
