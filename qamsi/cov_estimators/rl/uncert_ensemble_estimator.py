from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct

from qamsi.cov_estimators.rl.base_rl_estimator import BaseRLCovEstimator


class UncertEnsembleCovEstimator(BaseRLCovEstimator):
    def __init__(self, shrinkage_type: str, kernel=DotProduct()) -> None:
        super().__init__(shrinkage_type=shrinkage_type)

        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=12,
            random_state=12,
        )

        self.last_pred = None
        self._pred = None
        self.encountered_nan = False

        self.uncert = []
        self.uncert_w = []

        self.shrinkage_mean = None

    def _transform_shrinkage_target(self, shrinkage_target: pd.Series) -> pd.Series:
        self.shrinkage_mean = shrinkage_target.mean()
        shrinkage_target = shrinkage_target - self.shrinkage_mean
        return shrinkage_target

    def _inv_transform_shrinkage_target(self, shrinkage_target: float) -> float:
        return shrinkage_target + self.shrinkage_mean

    def _fit_shrinkage(
        self, features: pd.DataFrame, shrinkage_target: pd.Series
    ) -> None:
        pred = features["l_shrinkage_mu"].iloc[-1].item()

        if not np.isnan(pred):
            self._pred = pred
            self.last_pred = pred
        else:
            self._pred = self.last_pred

        shrinkage_target = self._transform_shrinkage_target(shrinkage_target)
        if shrinkage_target.isna().any():
            self.encountered_nan = True
        else:
            self.gpr.fit(X=features, y=shrinkage_target)
            self.encountered_nan = False

    def _predict_shrinkage(self, features: pd.DataFrame) -> float:
        if not self.encountered_nan:
            pred, sigma = self.gpr.predict(features, return_std=True)
            pred = self._inv_transform_shrinkage_target(pred.item())
            sigma = sigma.item()

            # sigma_w = (sigma - min(self.uncert)) / (max(self.uncert) - min(self.uncert)) if len(self.uncert) > 1 else 0
            sigma_w = sigma / np.mean(self.uncert) if len(self.uncert) > 1 else 1
            pred = sigma_w * pred
            self.uncert_w.append(sigma_w)
            # pred = sigma_w * self._pred + (1 - sigma_w) * pred

            self.uncert.append(sigma)

            pred = np.clip(pred, 0, 1)
            self.last_pred = pred
            return pred

        self.uncert.append(0)
        self.uncert_w.append(0)
        return self.last_pred
