from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso
# from scipy.stats import spearmanr

from qamsi.strategies.optimization_data import TrainingData, PredictionData
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator
from qamsi.features.covar import var_covar_from_corr_array_mac

warnings.filterwarnings("ignore")


class GLassoCovEstimator(BaseCovEstimator):
    START_SHRINKAGE: float = 0.3
    SHRINKAGE_STEP: float = 0.1

    def __init__(self) -> None:
        super().__init__()

        self._fitted_vols = None
        self._fitted_corr = None
        self._fitted_cov = None

    def _fit_glasso(
        self, correlation: np.ndarray, shrinkage: float = 0.3
    ) -> np.ndarray:
        try:
            reconstr_corr, _ = graphical_lasso(correlation, alpha=shrinkage)
            reconstr_corr = reconstr_corr.clip(min=-1, max=1)
            np.fill_diagonal(reconstr_corr, 1)
        except FloatingPointError:
            reconstr_corr = self._fit_glasso(
                correlation, shrinkage=shrinkage + self.SHRINKAGE_STEP
            )

        return reconstr_corr

    def _fit(self, training_data: TrainingData) -> None:
        ret = training_data.simple_excess_returns

        self._fitted_vols = np.eye(ret.shape[1]) * ret.std().to_numpy()
        # corr = spearmanr(ret).statistic
        corr = ret.corr().to_numpy()

        self._fitted_corr = self._fit_glasso(corr, shrinkage=self.START_SHRINKAGE)

        cov = var_covar_from_corr_array_mac(self._fitted_corr, self._fitted_vols)
        self._fitted_cov = pd.DataFrame(
            cov, index=self.available_assets, columns=self.available_assets
        )

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        return self._fitted_cov
