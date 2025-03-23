from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Optional
    import pandas as pd


class BaseStrategy(ABC):
    """An abstract base class representing a generic financial investment strategy.

    The Strategy class defines a template for implementing custom strategies
    with methods for fitting models, making predictions, and determining
    portfolio weights.

    """

    def __init__(
        self,
        random_seed: int | None = None,
        ml_metrics: Optional[list[Callable]] = None,
    ) -> None:
        super().__init__()
        self.random_seed = random_seed
        self.ml_metrics = ml_metrics

        self.all_assets = None
        self.available_assets = None

    def __call__(self, features: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        """Invoke the strategy to compute portfolio weights based on input features.

        Args:
            features (pd.DataFrame): A DataFrame containing the feature set used
            to compute portfolio weights.

        Returns:
            pd.DataFrame: A DataFrame of computed portfolio weights.

        """
        return self.get_weights(features=features, factors=factors)

    def fit(
        self, features: pd.DataFrame, factors: pd.DataFrame, targets: pd.DataFrame
    ) -> None:
        available_stocks = targets.loc[:, ~targets.iloc[-1].isna()].columns.tolist()
        self.all_assets = targets.columns.tolist()
        self.available_assets = targets[available_stocks].columns.tolist()

        self._fit(features=features, factors=factors, targets=targets[available_stocks])

    @abstractmethod
    def _fit(
        self, features: pd.DataFrame, factors: pd.DataFrame, targets: pd.DataFrame
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_weights(
        self, features: pd.DataFrame, factors: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate the portfolio weights based on the input features.

        Args:
            features (pd.DataFrame): A DataFrame containing the input feature variables.
            factors (pd.DataFrame): A DataFrame containing the input factor variables.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated portfolio weights.

        """
        raise NotImplementedError
