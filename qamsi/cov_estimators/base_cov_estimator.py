from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from qamsi.strategies.optimization_data import TrainingData, PredictionData

# import numpy as np


class BaseCovEstimator(ABC):
    def __init__(self) -> None:
        super().__init__()

        self._fitted = False
        self._available_assets = None

    @staticmethod
    def _check_positive_semi_definite(cov: pd.DataFrame) -> None:
        # assert np.all(np.linalg.eigvals(cov) >= 0), (
        #     "Covariance matrix is not positive semi-definite."
        # )
        pass

    def fit(self, training_data: TrainingData) -> None:
        self._fit(training_data=training_data)
        self._fitted = True

    def predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        assert self._fitted, "Model is not fitted yet! Call fit() first."

        pred_cov = self._predict(prediction_data=prediction_data)
        self._check_positive_semi_definite(pred_cov)
        return pred_cov

    def __call__(self, prediction_data: PredictionData) -> pd.DataFrame:
        return self.predict(prediction_data=prediction_data)

    @abstractmethod
    def _fit(self, training_data: TrainingData) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def available_assets(self) -> list[str]:
        return self._available_assets

    @available_assets.setter
    def available_assets(self, available_assets: list[str]) -> None:
        self._available_assets = available_assets
