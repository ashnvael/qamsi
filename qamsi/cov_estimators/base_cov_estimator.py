from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class BaseCovEstimator(ABC):
    def __init__(self) -> None:
        super().__init__()

        self._fitted = False

    @staticmethod
    def _check_positive_semi_definite(cov: pd.DataFrame) -> None:
        # assert np.all(np.linalg.eigvals(cov) >= 0), (
        #     "Covariance matrix is not positive semi-definite."
        # )
        pass

    def fit(
        self, features: pd.DataFrame, factors: pd.DataFrame, targets: pd.DataFrame
    ) -> None:
        self._fit(features, factors, targets)
        self._fitted = True

    def predict(self, features: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "Model is not fitted yet! Call fit() first."

        pred_cov = self._predict(features, factors)
        self._check_positive_semi_definite(pred_cov)
        return pred_cov

    def __call__(self, features: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        return self.predict(features, factors)

    @abstractmethod
    def _fit(
        self, features: pd.DataFrame, factors: pd.DataFrame, targets: pd.DataFrame
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict(self, features: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
