from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from qamsi.config.trading_config import TradingConfig


class TransactionCostCharger:
    def __init__(self, trading_config: TradingConfig) -> None:
        super().__init__()

        self.trading_config = trading_config

        self._strategy_total_r = None
        self._strategy_excess_r = None
        self._strategy_turnover = None

    @staticmethod
    def get_turnover(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        n_assets = weights.shape[1]

        weights_all = np.concatenate([np.zeros((1, n_assets)), weights], axis=0)
        returns_all = np.concatenate(
            [np.zeros((1, n_assets)), returns.to_numpy()], axis=0
        )

        period_end_weights = weights_all * (1 + returns_all)

        turnover = np.abs(weights.to_numpy() - period_end_weights[:-1]).sum(axis=1)

        return pd.Series(turnover, index=returns.index, name="turnover")

    def _get_trading_costs(
        self, weights: pd.DataFrame, returns: pd.DataFrame
    ) -> pd.Series:
        turnover = self.get_turnover(weights, returns)
        self._strategy_turnover = turnover

        trading_costs = np.where(
            turnover > 0,
            turnover * self.trading_config.ask_commission,
            np.abs(turnover) * self.trading_config.bid_commission,
        )
        trading_costs += (
            self.trading_config.broker_fee + self.trading_config.bid_ask_spread / 2
        ) * turnover

        # Assume selling at final point fully, i.e. trading_costs[-2] = trading_costs[-1] + trading_costs[-2]
        return pd.Series(trading_costs, index=turnover.index, name="trading_costs")

    def _get_success_costs(self, returns: pd.DataFrame) -> pd.Series:
        success_costs = pd.Series(
            np.zeros(len(returns)), index=returns.index, name="success_costs"
        )
        success_costs.iloc[-1] = (
            np.maximum(returns.add(1).prod().add(-1), 0)
            * self.trading_config.success_fee
        ).sum()
        return success_costs

    def _get_transaction_costs(
        self, weights: pd.DataFrame, returns: pd.DataFrame
    ) -> pd.Series:
        trading_costs = self._get_trading_costs(weights=weights, returns=returns)
        mf_costs = self._get_mf_costs(returns=returns)
        success_costs = self._get_success_costs(returns=returns)

        costs = pd.merge_asof(
            trading_costs,
            mf_costs,
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta("1D"),
        )
        costs = pd.merge_asof(
            costs,
            success_costs,
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta("1D"),
        )

        return costs.sum(axis=1)

    def _get_mf_costs(self, returns: pd.DataFrame) -> pd.Series:
        # TODO(Viacheslav Buchkov): fix to mf_freq parameterizable
        mf_costs = (
            pd.Series(
                np.zeros(returns.shape[0]), index=returns.index, name="management_fee"
            )
            .resample("YE")
            .sum()
        )
        mf_costs = mf_costs + self.trading_config.management_fee
        return mf_costs.add(1).cumprod().add(-1)

    def __call__(self, weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        return self._get_transaction_costs(weights, returns)
