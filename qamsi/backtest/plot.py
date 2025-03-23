from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def plot_cumulative_pnls(
    strategy_total: pd.Series,
    baseline: pd.Series,
    buy_hold: pd.Series | None = None,
    *,
    plot_log: bool = False,
) -> None:
    plt.figure(figsize=(14, 8))

    strategy_total.iloc[0, :] = np.zeros((1, strategy_total.shape[1]))
    baseline.iloc[0, :] = np.zeros((1, baseline.shape[1]))
    if buy_hold is not None:
        buy_hold.iloc[0, :] = np.zeros((1, buy_hold.shape[1]))

    strategy_cumulative = strategy_total.add(1).cumprod()
    baseline_cumulative = baseline.add(1).cumprod()

    strategy = (
        np.log(strategy_cumulative.to_numpy())
        if plot_log
        else strategy_cumulative.to_numpy()
    )
    baseline = (
        np.log(baseline_cumulative.to_numpy())
        if plot_log
        else baseline_cumulative.to_numpy()
    )

    plt.plot(strategy_cumulative.index, strategy, label="Strategy")
    plt.plot(baseline_cumulative.index, baseline, label="Baseline")

    if buy_hold is not None:
        buy_hold_cumulative = buy_hold.add(1).cumprod()
        for col in buy_hold.columns:
            buy_hold_series = (
                np.log(buy_hold_cumulative[col].to_numpy())
                if plot_log
                else buy_hold_cumulative[col].to_numpy()
            )
            plt.plot(buy_hold_cumulative.index, buy_hold_series, label=col)

    plt.xlabel("Date", fontsize=14)
    if plot_log:
        plt.ylabel("Log Cumulative Pnl", fontsize=14)
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

    plt.plot(turnover.index, turnover, label="Strategy Turnover")
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


def plot_histogram_vs_baseline(
    strategy_total: pd.Series,
    baseline: pd.Series,
) -> None:
    plt.figure(figsize=(14, 8))

    plt.hist(strategy_total, bins=50, alpha=0.5, label="Strategy Returns")
    plt.hist(baseline, bins=50, alpha=0.5, label="Baseline Returns")

    plt.ylabel("Return", fontsize=14)
    plt.legend(fontsize=16, loc="upper right")
    plt.show()
