from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradingConfig:
    # Exposures
    max_exposure: float = 1.0
    min_exposure: float = 0.0

    # Broker Fees
    broker_fee: float = 0.0
    # TODO @V: bid-ask in prices
    bid_ask_spread: float = 0.0  # For Mid-Term Strat assume const bid-ask
    ask_commission: float = 0.0
    bid_commission: float = 0.0

    # Fund Fees
    management_fee: float = 0.0
    success_fee: float = 0.0

    # Lag
    trading_days_lag: int = 1
