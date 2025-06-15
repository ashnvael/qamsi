from __future__ import annotations

import numpy as np
import pandas as pd

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

        self._predictions = []

    def _fit(self, training_data: TrainingData) -> None:
        self._seen_training_data = training_data

        feat = training_data.features["cgp_ucb"]

        if feat.isna().all():
            if self.last_pred is not None:
                pred = self.last_pred
            else:
                pred = training_data.features["target_rolling_mean"].iloc[-1].item()
        else:
            last_valid_index = feat.last_valid_index()
            last_valid_value = (
                feat.loc[last_valid_index] if last_valid_index is not None else None
            )

            pred = last_valid_value.item()

        if not np.isnan(pred):
            self._pred = pred
            self.last_pred = pred
        else:
            self._pred = self.last_pred

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        self.alpha = self._pred

        super()._fit(training_data=self._seen_training_data)

        self._seen_training_data = None

        self._predictions.append([prediction_data.features.index[-1], self._pred])

        return self._fitted_cov

    @property
    def predictions(self) -> pd.DataFrame | None:
        if self._predictions is None:
            return None

        pred = pd.DataFrame(self._predictions, columns=["date", "prediction"])
        pred["date"] = pd.to_datetime(pred["date"])
        return pred.set_index("date")
