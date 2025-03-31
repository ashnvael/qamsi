from __future__ import annotations

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator


class GBMPredictor(BaseCovEstimator):
    def __init__(self) -> None:
        super().__init__()

        self._models = {}
        self._available_assets = None
        self._obs_means = {}

    def _fit(
        self, features: pd.DataFrame, factors: pd.DataFrame, targets: pd.DataFrame
    ) -> None:
        self._available_assets = list(targets.columns)
        for stock in self._available_assets:
            self._models[stock] = RandomForestRegressor(
                n_estimators=30, min_samples_leaf=5, random_state=12
            )
            X = self._update_features(features.iloc[1:], factors.iloc[1:], targets[stock].shift(1).iloc[1:])
            self._models[stock].fit(X, targets[stock] ** 2)
            self._obs_means[stock] = targets[stock].mean()

    def _predict(self, features: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        available_assets = self._available_assets
        covmat = pd.DataFrame(index=available_assets, columns=available_assets)
        for stock in available_assets:
            X = self._update_features(features, factors)
            covmat.loc[stock, stock] = self._models[stock].predict(X) - self._obs_means[stock] ** 2

        return covmat.clip(lower=0).astype(float)

    @staticmethod
    def _update_features(features: pd.DataFrame, factors: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        return pd.concat([features, factors, target], axis=1)
