from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def plot_cumulative_pnls(
    strategy_total: pd.Series,
    buy_hold: pd.Series | None = None,
    *,
    plot_log: bool = False,
    name_strategy: str = "Strategy",
    mkt_name: str = "spx",
) -> None:
    plt.figure(figsize=(14, 8))

    strategy_total = strategy_total.copy()
    buy_hold = buy_hold.copy() if buy_hold is not None else None

    strategy_total.iloc[0, :] = np.zeros((1, strategy_total.shape[1]))
    if buy_hold is not None:
        buy_hold.iloc[0, :] = np.zeros((1, buy_hold.shape[1]))

    strategy_cumulative = strategy_total.add(1).cumprod()
    strategy = strategy_cumulative.to_numpy()

    plt.plot(strategy_cumulative.index, strategy, label=name_strategy)

    if buy_hold is not None:
        buy_hold_cumulative = buy_hold.add(1).cumprod()
        if mkt_name in buy_hold.columns:
            market = buy_hold_cumulative[mkt_name]
            buy_hold_cumulative = buy_hold_cumulative.drop(columns=mkt_name)
            plt.plot(market.index, market.to_numpy(), label=mkt_name, linewidth=6)
        plt.plot(
            buy_hold_cumulative.index,
            buy_hold_cumulative.to_numpy(),
            label=buy_hold_cumulative.columns,
            linestyle="--",
        )
        plt.legend(
            fontsize=16,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0.0,
        )

    plt.xlabel("Date", fontsize=14)
    if plot_log:
        plt.ylabel("Log Scale Cumulative Pnl", fontsize=14)
        plt.yscale("log")
    else:
        plt.ylabel("Cumulative Pnl", fontsize=14)
    plt.legend(fontsize=16)
    plt.show()


def plot_weights(weights: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 8))

    plt.plot(weights.index, weights.to_numpy(), label="Strategy Weights")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Weight", fontsize=14)
    plt.legend(fontsize=16)
    plt.show()


def plot_turnover(turnover: pd.Series) -> None:
    plt.figure(figsize=(14, 8))
    dates = turnover.index
    turnover = turnover.copy().to_numpy()

    rebal_turnober_idx = turnover != 0
    turnover = turnover[rebal_turnober_idx]
    dates = dates[rebal_turnober_idx]

    plt.plot(dates, turnover, label="Strategy Turnover")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Turnover", fontsize=14)
    plt.legend(fontsize=16)
    plt.show()


def plot_histogram(
    strategy_total: pd.Series,
) -> None:
    plt.figure(figsize=(14, 8))

    plt.hist(strategy_total, bins=50, label="Strategy Returns")

    plt.ylabel("Return", fontsize=14)
    plt.legend(fontsize=16)
    plt.show()


def plot_outperformance(
    strategy_total: pd.Series,
    baseline: pd.Series,
    baseline_name: str = "Baseline",
) -> None:
    dates = strategy_total.index

    strategy_total_r = strategy_total.to_numpy().flatten()
    baseline_total_r = baseline.to_numpy().flatten()

    outperform = strategy_total_r - baseline_total_r
    outperform_rel = outperform / (1 + baseline_total_r)

    plt.figure(figsize=(14, 8))
    plt.plot(dates, np.log(1 + outperform_rel).cumsum())

    plt.xlabel("Date")
    plt.ylabel("Outperformance")
    plt.title(f"Outperformance vs {baseline_name}")

    plt.show()
