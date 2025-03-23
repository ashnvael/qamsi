from __future__ import annotations

import numpy as np
import pandas as pd
from qamsi.strategies.base_strategy import BaseStrategy


class EWStrategy(BaseStrategy):
    def __init__(
        self, rebal_freq: str | None = None, random_seed: int | None = None
    ) -> None:
        super().__init__()
        self.rebal_freq = rebal_freq
        self.random_seed = random_seed

        self.all_assets = None
        self.available_assets = None

    def _fit(
        self, features: pd.DataFrame, factors: pd.DataFrame, targets: pd.DataFrame
    ) -> None:
        pass

    def get_weights(
        self, features: pd.DataFrame, factors: pd.DataFrame
    ) -> pd.DataFrame:
        n_assets = len(self.available_assets)
        weights = pd.DataFrame(0.0, index=[features.index[-1]], columns=self.all_assets)
        weights.loc[:, self.available_assets] = (
            np.ones((1, n_assets), dtype=float) / n_assets
        )

        return weights
