from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from qamsi.betas_estimators.base_betas_estimator import BaseBetasEstimator

class LinearBetasEstimator(BaseBetasEstimator):
    """
    Computes factor loadings via linear regression of targets on factors.
    """
    def __init__(self) -> None:
        super().__init__()
        self._factor_exposures: pd.DataFrame | None = None
        self._residuals: pd.DataFrame | None = None

    def _fit(
        self,
        features: pd.DataFrame,
        factors: pd.DataFrame,
        targets: pd.DataFrame
    ) -> None:
        exposures: list[pd.Series] = []
        residuals: list[pd.Series] = []
        for asset in targets.columns:

            X = sm.add_constant(factors)
            y = targets[asset]
            model = sm.OLS(y, X)
            results = model.fit()

            exposures.append(results.params[factors.columns])
            residuals.append(results.resid)

        self._factor_exposures = pd.DataFrame(
            exposures,
            index=targets.columns,
            columns=factors.columns
        )

        self._residuals = pd.DataFrame(
            residuals,
            index=targets.columns,
            columns=targets.index
        ).T

    def _predict(
        self,
        features: pd.DataFrame,
        factors: pd.DataFrame
    ) -> pd.DataFrame:
        assert self._factor_exposures is not None, "Estimator must be fitted before predicting."
        return self._factor_exposures