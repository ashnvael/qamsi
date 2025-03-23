from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from qamsi.hedge.hedger import Hedger


class MarketFuturesHedge(Hedger):
    def __init__(self) -> None:
        super().__init__()
        self._betas = None
        self.hedge_assets = None

    def fit(
        self,
        features: pd.DataFrame,
        rf_rate: pd.DataFrame,
        hedge_assets: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> None:  # noqa: ARG002
        self.hedge_assets = hedge_assets.columns
        self._betas = self._get_betas(
            market_index=hedge_assets, rf_rate=rf_rate, targets=targets
        )

    @staticmethod
    def _get_betas(
        market_index: pd.DataFrame, rf_rate: pd.DataFrame, targets: pd.DataFrame
    ):  # noqa: ANN205
        erp = market_index - rf_rate

        betas = []
        for stock in targets.columns:
            xs_r = targets[stock] - rf_rate.flatten()

            lr = sm.OLS(xs_r, erp).fit()

            betas.append([stock, lr.params.to_numpy()[0]])

        betas = pd.DataFrame(betas, columns=["stock", "beta"])
        return betas.set_index("stock")

    def get_weights(
        self, features: pd.DataFrame, weights: pd.DataFrame
    ) -> pd.DataFrame:
        hedge_weights = -np.nansum(
            self._betas.to_numpy().T * weights.to_numpy(), axis=1
        )

        return pd.DataFrame(
            hedge_weights, index=features.index, columns=self.hedge_assets
        )

    @property
    def betas(self) -> pd.DataFrame:
        return self._betas
