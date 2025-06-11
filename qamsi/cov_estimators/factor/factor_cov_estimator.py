from __future__ import annotations

from copy import deepcopy

import pandas as pd

from qamsi.strategies.optimization_data import TrainingData, PredictionData
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator
from qamsi.features.ols_betas import get_exposures


class FactorCovEstimator(BaseCovEstimator):
    def __init__(
        self,
        factor_cov_estimator: BaseCovEstimator,
        residual_cov_estimator: BaseCovEstimator,
        factors_selection: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.factor_cov_estimator = factor_cov_estimator
        self.residual_cov_estimator = residual_cov_estimator
        self.factors_selection = factors_selection

        self._factor_exposures = None
        self._residuals = None

    def _fit(self, training_data: TrainingData) -> None:
        factors = training_data.factors

        if self.factors_selection is not None:
            factors = factors[self.factors_selection]

        self._factor_exposures, self._residuals = get_exposures(
            factors=factors,
            targets=training_data.simple_excess_returns,
            return_residuals=True,
        )

        factor_train_data = deepcopy(training_data)
        factor_train_data.simple_excess_returns = factors
        factor_train_data.log_excess_returns = None

        resid_train_data = deepcopy(training_data)
        resid_train_data.simple_excess_returns = self._residuals
        resid_train_data.log_excess_returns = None

        self.factor_cov_estimator.fit(factor_train_data)
        self.residual_cov_estimator.fit(resid_train_data)

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        factor_cov = (
            self.factor_cov_estimator.predict(prediction_data).to_numpy().astype(float)
        )
        residual_cov = (
            self.residual_cov_estimator.predict(prediction_data)
            .to_numpy()
            .astype(float)
        )

        exposures = self._factor_exposures.to_numpy()

        covmat = exposures @ factor_cov @ exposures.T + residual_cov
        covmat = pd.DataFrame(
            covmat, index=self._available_assets, columns=self._available_assets
        )

        return covmat.astype(float)
