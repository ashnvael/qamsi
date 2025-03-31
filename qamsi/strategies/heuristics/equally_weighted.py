from __future__ import annotations

import numpy as np
import pandas as pd
from qamsi.strategies.base_strategy import BaseStrategy


class EWStrategy(BaseStrategy):
    def __init__(self, rebalance_mode: str = "fully") -> None:
        super().__init__(rebalance_mode=rebalance_mode)

        self.all_assets = None
        self.available_assets = None

    def _fit(
        self, features: pd.DataFrame, factors: pd.DataFrame, targets: pd.DataFrame
    ) -> None:
        pass

    def _get_weights(
        self, features: pd.DataFrame, factors: pd.DataFrame
    ) -> pd.DataFrame:
        n_assets = len(self.available_assets)
        weights = pd.DataFrame(0.0, index=[features.index[-1]], columns=self.all_assets)
        weights.loc[:, self.available_assets] = (
            np.ones((1, n_assets), dtype=float) / n_assets
        )

        return weights
