from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator


def corr_matrix_from_cov(var_covar: np.ndarray) -> np.ndarray:
    diag_inv = np.diag(1 / np.sqrt(np.diag(var_covar)))
    return diag_inv @ var_covar @ diag_inv


def var_covar_from_corr_array(corr_array: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
    return volatilities @ corr_array @ volatilities


class PCACovEstimator(BaseCovEstimator):
    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self.k = k

        self._fitted_vols = None
        self._fitted_corr = None
        self._fitted_cov = None

        self._obs_cov = None

        self._pca = PCA()

    def _fit(
        self, features: pd.DataFrame, factors: pd.DataFrame, targets: pd.DataFrame
    ) -> None:
        self._available_assets = list(targets.columns)
        self._obs_cov = targets.cov()
        self._fitted_vols = np.eye(targets.shape[1]) * targets.std().to_numpy()
        corr = targets.corr().to_numpy()

        self._pca.fit(corr)

        components = self._pca.components_
        components = components[:self.k, :]

        reduced_data = self._pca.transform(corr)[:, :self.k]

        reconstr_corr = reduced_data @ components + self._pca.mean_
        self._fitted_corr = reconstr_corr.clip(min=-1, max=1)
        np.fill_diagonal(self._fitted_corr, 1)

        cov = var_covar_from_corr_array(self._fitted_corr, self._fitted_vols)
        self._fitted_cov = pd.DataFrame(cov, index=self._available_assets, columns=self._available_assets)

    def _predict(self, features: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        return self._fitted_cov
