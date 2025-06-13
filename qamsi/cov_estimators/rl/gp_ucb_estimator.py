from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import numpy as np

from qamsi.cov_estimators.shrinkage.rp_cov_estimator import RiskfolioCovEstimator
from qamsi.strategies.optimization_data import PredictionData, TrainingData


class GPUCBCovEstimator(RiskfolioCovEstimator):
    def __init__(self) -> None:
        super().__init__(
            estimator_type="shrunk",
            alpha=0.1,
        )

        self._pred = None
        self.last_pred = None

    def _fit(self, training_data: TrainingData) -> None:
        self._seen_training_data = training_data

        pred = training_data.features["gp_ucb_pred"].iloc[-1].item()

        if not np.isnan(pred):
            self._pred = pred
            self.last_pred = pred
        else:
            self._pred = self.last_pred

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        self.alpha = self._pred

        super()._fit(training_data=self._seen_training_data)

        self._seen_training_data = None

        return self._fitted_cov
