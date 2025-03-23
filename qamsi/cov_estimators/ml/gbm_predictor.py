from __future__ import annotations

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator


class GBMPredictor(BaseCovEstimator):
    def __init__(self) -> None:
        super().__init__()

        self._models = {}

    def _fit(
        self, features: pd.DataFrame, factors: pd.DataFrame, targets: pd.DataFrame
    ) -> None:
        for stock in targets.columns:
            self._models[stock] = GradientBoostingRegressor(
                n_estimators=30, max_depth=5, random_state=12
            )
            self._models[stock].fit(features, targets[stock] ** 2)

    def _predict(self, features: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        stocks = list(self._models.keys())
        covmat = pd.DataFrame(index=stocks, columns=stocks)
        for stock in stocks:
            covmat.loc[stock, stock] = self._models[stock].predict(features)

        return covmat.clip(lower=0)
