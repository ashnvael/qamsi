from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class RebalanceMode(Enum):
    FULLY = "fully"
    ONLY_CHANGED = "only_changed"


class BaseStrategy(ABC):
    """An abstract base class representing a generic financial investment strategy.

    The Strategy class defines a template for implementing custom strategies
    with methods for fitting models, making predictions, and determining
    portfolio weights.

    """

    def __init__(self, rebalance_mode: str = "fully") -> None:
        super().__init__()
        self.rebalance_mode = RebalanceMode(rebalance_mode)

        self.all_assets = None
        self.available_assets = None

        self._past_weights = None

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

    def get_weights(
        self, features: pd.DataFrame, factors: pd.DataFrame
    ) -> pd.DataFrame:
        weights = self._get_weights(features=features, factors=factors)

        if self.rebalance_mode == RebalanceMode.FULLY:
            pass
        elif self.rebalance_mode == RebalanceMode.ONLY_CHANGED:
            assert (weights.to_numpy() >= 0).all(), (
                f"Weights must be non-negative for the method {self.rebalance_mode}."
            )  # noqa: S101
            if self._past_weights is not None:
                new_weights = weights.to_numpy().flatten()
                past_weights = self._past_weights.to_numpy().flatten()
                proposed_change = new_weights - past_weights

                were_zero_mask = np.where(past_weights == 0, 1, 0)
                will_be_non_zero = np.where(new_weights != 0, 1, 0)

                were_non_zero_mask = np.where(past_weights != 0, 1, 0)
                want_set_zero_mask = np.where(new_weights == 0, 1, 0)

                accept_change = (were_zero_mask & will_be_non_zero) | (
                    were_non_zero_mask & want_set_zero_mask
                )

                adj_weights = past_weights + proposed_change * accept_change
                adj_weights = adj_weights / adj_weights.sum()
            else:
                adj_weights = weights.to_numpy().flatten()

            weights = pd.DataFrame(
                adj_weights[np.newaxis, :], index=weights.index, columns=weights.columns
            )
        else:
            msg = f"RebalanceMode {self.rebalance_mode} not implemented."
            raise NotImplementedError(msg)

        self._past_weights = weights
        return weights

    @abstractmethod
    def _fit(
        self, features: pd.DataFrame, factors: pd.DataFrame, targets: pd.DataFrame
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _get_weights(
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
