from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator


class FactorPredictor(BaseCovEstimator):
    def __init__(self) -> None:
        super().__init__()

        self._obs_squared_r = {}
        self._available_assets = None
        self._obs_means = {}
        self._obs_corr = None
        self._obs_cov = None

    def _fit(
        self, features: pd.DataFrame, factors: pd.DataFrame, targets: pd.DataFrame
    ) -> None:
        self._available_assets = list(targets.columns)
        self._obs_corr = targets.corr().to_numpy()
        self._obs_cov = targets.cov()
        for stock in self._available_assets:
            self._obs_squared_r[stock] = (targets[stock] ** 2).mean()
            self._obs_means[stock] = targets[stock].mean()

    def _predict(self, features: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        available_assets = self._available_assets
        vols = []
        for stock in available_assets:
            vols.append(self._obs_squared_r[stock] - self._obs_means[stock] ** 2)

        vols = np.eye(len(vols)) * np.sqrt(np.array(vols))
        cov = self.var_covar_from_corr_array(self._obs_corr, vols)
        covmat = pd.DataFrame(cov, index=available_assets, columns=available_assets)

        return covmat.astype(float)

    @staticmethod
    def var_covar_from_corr_array(
            corr_array: np.ndarray, volatilities: np.ndarray = None
    ) -> np.ndarray:
        if volatilities is None:
            volatilities = np.ones_like(corr_array)
        return volatilities @ corr_array @ volatilities
