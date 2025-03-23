from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class Hedger(ABC):
    def __init__(self, random_seed: int | None = None) -> None:
        super().__init__()
        self.random_seed = random_seed

    def __call__(self, features: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
        return self.get_weights(features=features, weights=weights)

    @abstractmethod
    def get_weights(
        self, features: pd.DataFrame, weights: pd.DataFrame
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def fit(
        self,
        features: pd.DataFrame,
        rf_rate: pd.DataFrame,
        hedge_assets: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> None:
        raise NotImplementedError
