from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from qamsi.strategies.optimization_data import TrainingData, PredictionData
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator
from qamsi.features.covar import var_covar_from_corr_array_mac


class PCACovEstimator(BaseCovEstimator):
    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self.k = k

        self._fitted_vols = None
        self._fitted_corr = None
        self._fitted_cov = None

        self._obs_cov = None

        self._pca = PCA()

    def _fit(self, training_data: TrainingData) -> None:
        ret = training_data.simple_excess_returns

        self._obs_cov = ret.cov()
        self._fitted_vols = np.eye(ret.shape[1]) * ret.std().to_numpy()
        corr = ret.corr().to_numpy()

        self._pca.fit(corr)

        components = self._pca.components_
        components = components[: self.k, :]

        reduced_data = self._pca.transform(corr)[:, : self.k]

        reconstr_corr = reduced_data @ components + self._pca.mean_
        self._fitted_corr = reconstr_corr.clip(min=-1, max=1)
        np.fill_diagonal(self._fitted_corr, 1)

        cov = var_covar_from_corr_array_mac(self._fitted_corr, self._fitted_vols)
        self._fitted_cov = pd.DataFrame(
            cov, index=self.available_assets, columns=self.available_assets
        )

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        return self._fitted_cov
