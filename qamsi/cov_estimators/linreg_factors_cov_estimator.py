from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator


class LinregFactorsCovEstimator(BaseCovEstimator):
    def __init__(
        self,
        factor_cov_estimator: BaseCovEstimator,
        residual_cov_estimator: BaseCovEstimator,
    ) -> None:
        super().__init__()

        self.factor_cov_estimator = factor_cov_estimator
        self.residual_cov_estimator = residual_cov_estimator

        self._factor_exposures = None
        self._residuals = None

    def _fit_factor_exposures(
        self, factors: pd.DataFrame, targets: pd.DataFrame
    ) -> None:
        exposures = []
        residuals = []
        for stock in targets.columns:
            X = sm.add_constant(factors)
            y = targets[stock]
            model = sm.OLS(y, X)
            results = model.fit()

            exposures.append(results.params)
            residuals.append(results.resid)

        self._factor_exposures = pd.DataFrame(
            exposures, index=targets.columns, columns=factors.columns
        )
        self._residuals = pd.DataFrame(
            residuals, index=targets.columns, columns=targets.index
        ).T

    def _fit(
        self, features: pd.DataFrame, factors: pd.DataFrame, targets: pd.DataFrame
    ) -> None:
        self._fit_factor_exposures(factors, targets)

        self.factor_cov_estimator.fit(features, factors, factors)
        self.residual_cov_estimator.fit(features, factors, self._residuals)

    def _predict(self, features: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        factor_cov = (
            self.factor_cov_estimator.predict(features, factors)
            .to_numpy()
            .astype(float)
        )
        residual_cov = self.residual_cov_estimator.predict(features, factors).to_numpy().astype(float)

        exposures = self._factor_exposures.to_numpy()

        return exposures @ factor_cov @ exposures.T + residual_cov
