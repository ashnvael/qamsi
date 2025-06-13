from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradingConfig:
    # Exposures
    max_exposure: float | None = None
    min_exposure: float | None = None
    total_exposure: float | None = None

    # Broker Fees
    broker_fee: float = 0.0
    # TODO @V: bid-ask in prices
    bid_ask_spread: float = 0.0  # For Mid-Term Strat assume const bid-ask
    ask_commission: float = 0.0
    bid_commission: float = 0.0

    # Fund Fees
    management_fee: float = 0.0
    success_fee: float = 0.0

    # Trading Setup
    trading_lag_days: int | None = 1
