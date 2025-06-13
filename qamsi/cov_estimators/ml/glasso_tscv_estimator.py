from __future__ import annotations

import warnings

import numpy as np
from sklearn.covariance import GraphicalLassoCV
from sklearn.model_selection import TimeSeriesSplit

from qamsi.cov_estimators.ml.glasso_estimator import GLassoCovEstimator

warnings.filterwarnings("ignore")


class GLassoTSCVCovEstimator(GLassoCovEstimator):
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
            alphas = np.logspace(-1.5, 1, num=10)
            gl = GraphicalLassoCV(
                alphas=alphas,
                cv=TimeSeriesSplit(n_splits=10),
            )
            gl.fit(correlation)
            reconstr_corr = gl.covariance_
            reconstr_corr = reconstr_corr.clip(min=-1, max=1)
            np.fill_diagonal(reconstr_corr, 1)
        except FloatingPointError:
            reconstr_corr = self._fit_glasso(
                correlation, shrinkage=shrinkage + self.SHRINKAGE_STEP
            )

        return reconstr_corr
