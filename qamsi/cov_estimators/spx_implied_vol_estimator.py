from __future__ import annotations

import pandas as pd
import numpy as np

from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator
from qamsi.betas_estimators.base_betas_estimator import BaseBetasEstimator


class SPXSingleFactorImpliedVolCovEstimator(BaseCovEstimator):
    """
    Single-factor covariance estimator using only the SPX market factor.
    The factor variance is set to the implied volatility squared from features.

    Cov_f = [[ implied_vol^2 ]]
    """
    def __init__(
        self,
        factor_name: str = "spx",
        ivol_feature_name: str = "spx_implied_vol",
        use_last: bool = True
    ) -> None:
        super().__init__()
        self.factor_name = factor_name
        self.ivol_feature_name = ivol_feature_name
        self.use_last = use_last
        self._factor_cov: pd.DataFrame | None = None

    def _fit(
        self,
        features: pd.DataFrame,
        factors: pd.DataFrame,
        targets: pd.DataFrame
    ) -> None:

        ivol_series = features[self.ivol_feature_name]
        vol_value = ivol_series.iloc[-1] if self.use_last else ivol_series.mean()
        implied_var = vol_value ** 2

        self._factor_cov = pd.DataFrame(
            [[implied_var]],
            index=[self.factor_name],
            columns=[self.factor_name]
        )

    def _predict(
        self,
        features: pd.DataFrame,
        factors: pd.DataFrame
    ) -> pd.DataFrame:
        assert self._factor_cov is not None, "Estimator not fitted: call fit() first."
        return self._factor_cov.copy()


class SPXIVolFactorCovEstimator(BaseCovEstimator):
    """
    Combined factor-covariance estimator for a single SPX market factor using implied volatility.
    """
    def __init__(
        self,
        betas_estimator: BaseBetasEstimator,
        residual_cov_estimator: BaseCovEstimator,
        factor_name: str = "spx",
        ivol_feature_name: str = "spx_implied_vol",
        use_last: bool = True
    ) -> None:
        super().__init__()
        self.betas_estimator = betas_estimator
        self.factor_cov_estimator = SPXSingleFactorImpliedVolCovEstimator(
            factor_name=factor_name,
            ivol_feature_name=ivol_feature_name,
            use_last=use_last
        )
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
        assert self._factor_exposures is not None and self._residuals is not None, \
            "Model is not fitted yet. Call fit() first."

        factor_cov_df = self.factor_cov_estimator.predict(features, factors)
        factor_cov = factor_cov_df.to_numpy().astype(float)
        residual_cov = self.residual_cov_estimator.predict(features, factors).to_numpy().astype(float)
        exposures = self._factor_exposures.to_numpy().astype(float)
        total_cov = exposures @ factor_cov @ exposures.T + residual_cov
        assets = self._factor_exposures.index
        return pd.DataFrame(total_cov, index=assets, columns=assets)
