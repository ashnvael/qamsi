from __future__ import annotations

import numpy as np
import pandas as pd

from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator
from qamsi.betas_estimators.base_betas_estimator import BaseBetasEstimator

class FactorCovEstimator(BaseCovEstimator):
    """
    Factor-based covariance estimator Cov = B Σ_f B' + Σ_resid
    """
    def __init__(
        self,
        betas_estimator: BaseBetasEstimator,
        factor_cov_estimator: BaseCovEstimator,
        residual_cov_estimator: BaseCovEstimator,
    ) -> None:
        super().__init__()
        self.betas_estimator = betas_estimator
        self.factor_cov_estimator = factor_cov_estimator
        self.residual_cov_estimator = residual_cov_estimator
        self._factor_exposures: pd.DataFrame | None = None
        self._residuals: pd.DataFrame | None = None

    def _fit(
        self,
        features: pd.DataFrame,
        factors: pd.DataFrame,
        targets: pd.DataFrame
    ) -> None:

        self.betas_estimator.fit(features, factors, targets)
        self._factor_exposures = self.betas_estimator.predict(features, factors)
        self._residuals = getattr(self.betas_estimator, "_residuals")

        self.factor_cov_estimator.fit(features, factors, factors)
        self.residual_cov_estimator.fit(features, factors, self._residuals)

    def _predict(
        self,
        features: pd.DataFrame,
        factors: pd.DataFrame
    ) -> pd.DataFrame:
        assert self._factor_exposures is not None and self._residuals is not None, "Model is not fitted yet. Call fit() first."

        factor_cov = self.factor_cov_estimator.predict(features, factors).to_numpy().astype(float)
        residual_cov = self.residual_cov_estimator.predict(features, factors).to_numpy().astype(float)
        exposures = self._factor_exposures.to_numpy()

        total_cov = exposures @ factor_cov @ exposures.T + residual_cov
        assets = self._factor_exposures.index
        return pd.DataFrame(total_cov, index=assets, columns=assets)