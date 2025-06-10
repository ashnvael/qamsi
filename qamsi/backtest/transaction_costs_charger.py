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
        period_end_weights = [np.zeros(weights.shape[1])]
        last_rebal_date = weights.index[0]
        for rebal in [*weights.index[1:-1], None]:
            w0 = weights.loc[last_rebal_date]
            end_date = rebal - pd.Timedelta(days=1) if rebal else None
            ret_sample = returns.loc[last_rebal_date:end_date].copy().fillna(0)
            ret_sample = ret_sample.prod(axis=0).to_numpy()
            period_end_weights.append(w0 * (1 + ret_sample))
            last_rebal_date = rebal

        period_end_weights = np.array(period_end_weights)

        turnover = np.abs(weights.to_numpy() - period_end_weights).sum(axis=1)
        if len(weights) > 1:
            turnover = pd.Series(turnover, index=weights.index, name="turnover", dtype=np.float64)
        else:
            turnover = pd.Series(turnover[:-1], index=weights.index, name="turnover", dtype=np.float64)

        return turnover  # type: ignore[no-any-return]

    def _get_trading_costs(self, weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        turnover = self.get_turnover(weights, returns)
        self._strategy_turnover = turnover

        trading_costs = pd.Series(index=returns.index, name="trading_costs", dtype=np.float64)
        # TODO(@V): Bid and Ask commission
        tc = (self.trading_config.broker_fee + self.trading_config.bid_ask_spread / 2) * turnover
        trading_costs.loc[tc.index] = tc

        return trading_costs

    def _get_success_costs(self, returns: pd.DataFrame) -> pd.Series:
        success_costs = pd.Series(np.zeros(len(returns)), index=returns.index, name="success_costs")
        success_costs.iloc[-1] = (np.maximum(returns.add(1).prod().add(-1), 0) * self.trading_config.success_fee).sum()
        return success_costs

    def _get_transaction_costs(self, weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        trading_costs = self._get_trading_costs(weights=weights, returns=returns)
        mf_costs = self._get_mf_costs(returns=returns)
        success_costs = self._get_success_costs(returns=returns)

        costs = pd.merge_asof(trading_costs, mf_costs, left_index=True, right_index=True, tolerance=pd.Timedelta("1D"))
        costs = pd.merge_asof(costs, success_costs, left_index=True, right_index=True, tolerance=pd.Timedelta("1D"))

        return costs.sum(axis=1)

    def _get_mf_costs(self, returns: pd.DataFrame) -> pd.Series:
        # TODO(V): fix to mf_freq parameterizable
        mf_costs = pd.Series(np.zeros(returns.shape[0]), index=returns.index, name="management_fee").resample("YE").sum()
        mf_costs = mf_costs + self.trading_config.management_fee
        return mf_costs.add(1).cumprod().add(-1)

    def __call__(self, weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        return self._get_transaction_costs(weights, returns)

    @property
    def turnover(self) -> pd.Series:
        return self._strategy_turnover
