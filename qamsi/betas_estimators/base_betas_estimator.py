from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

class BaseBetasEstimator(ABC):
    """
    Abstract base class for estimating factor loadings (betas).
    """
    def __init__(self) -> None:
        super().__init__()
        self._fitted = False

    def fit(
        self,
        features: pd.DataFrame,
        factors: pd.DataFrame,
        targets: pd.DataFrame
    ) -> None:
        """
        Fit the loadings estimator to compute loadings (and residuals, if any).
        """
        self._fit(features, factors, targets)
        self._fitted = True

    def predict(
        self,
        features: pd.DataFrame,
        factors: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Return estimated factor loadings as a DataFrame of shape (n_assets, n_factors).
        """
        assert self._fitted, "Loadings estimator is not fitted yet! Call fit() first."
        return self._predict(features, factors)

    def __call__(
        self,
        features: pd.DataFrame,
        factors: pd.DataFrame
    ) -> pd.DataFrame:
        return self.predict(features, factors)

    @abstractmethod
    def _fit(
        self,
        features: pd.DataFrame,
        factors: pd.DataFrame,
        targets: pd.DataFrame
    ) -> None:
        """
        Internal fit method to be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def _predict(
        self,
        features: pd.DataFrame,
        factors: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Internal predict method to be implemented by subclasses.
        """
        raise NotImplementedError