from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import numpy as np

from qamsi.cov_estimators.rl.base_rl_estimator import BaseRLCovEstimator


class MALinearCovEstimator(BaseRLCovEstimator):
    def __init__(self, shrinkage_type: str) -> None:
        super().__init__(shrinkage_type=shrinkage_type)

        self._pred = None
        self.last_pred = None

    def _fit_shrinkage(
        self, features: pd.DataFrame, shrinkage_target: pd.Series
    ) -> None:
        pred = features["target_rolling_mean"].iloc[-1].item()

        if not np.isnan(pred):
            self._pred = pred
            self.last_pred = pred
        else:
            self._pred = self.last_pred

    def _predict_shrinkage(self, features: pd.DataFrame) -> float:
        return self._pred
