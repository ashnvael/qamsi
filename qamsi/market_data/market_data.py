from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime as dt

    import pandas as pd

    from qamsi.market_data.ticker import Ticker

from functools import cache

import numpy as np
import yfinance as yfin

from qamsi.market_data.tickers import Tickers


class MarketData:
    TARGET_COLUMN = "Adj Close"

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        sampling_period: None | str = "D",
        tickers: list[Ticker] | Tickers | None = None,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
    ) -> None:
        self.data = data
        self.sampling_period = sampling_period

        if isinstance(tickers, list):
            self.tickers = Tickers(tickers)
        elif isinstance(tickers, Tickers):
            self.tickers = tickers
        else:
            msg = "tickers must be a list[Ticker] or Tickers instance"
            raise ValueError(msg)  # noqa: TRY004

        self.start = start
        self.end = end

        self._initialize()

    def __len__(self) -> int:
        return len(self.tickers)

    def _initialize(self) -> None:
        if self.data is None:
            self._load_yahoo()

        if self.sampling_period:
            self._resample_data()
        self._create_returns_dfs()

    def _load_yahoo(self) -> None:
        self.data = yfin.download(self.tickers.codes, self.start, self.end)[
            self.TARGET_COLUMN
        ]
        if self.data is None:
            msg = "Failed to load data from Yahoo Finance"
            raise ValueError(msg)
        if self.data.shape[1] != len(self.tickers.codes):
            for i, ticker in enumerate(self.tickers.codes):
                if i < self.data.shape[1]:
                    if self.data.columns[i] != ticker:
                        self.data.insert(
                            i, column=f"{ticker}_2", value=self.data.loc[:, ticker]
                        )
                else:
                    self.data.insert(
                        i, column=f"{ticker}_2", value=self.data.loc[:, ticker]
                    )

    def _resample_data(self) -> None:
        if self.data is None:
            self._load_yahoo()
            if self.data is None:
                msg = "Failed to load data in _resample_data"
                raise ValueError(msg)
        if self.sampling_period is not None:
            self.data = self.data.resample(self.sampling_period).first()

    def _create_returns_dfs(self) -> None:
        if self.data is None:
            msg = "Data is None"
            raise ValueError(msg)
        self._df_log_returns = (self.data / self.data.shift(1)).apply(np.log)
        self._df_simple_returns = self.data / self.data.shift(1) - 1

    @property
    def log_returns(self) -> pd.DataFrame:
        return self._df_log_returns

    @property
    def simple_returns(self) -> pd.DataFrame:
        return self._df_simple_returns

    def plot(self) -> None:
        if self.data is None:
            msg = "Data is None"
            raise ValueError(msg)

        n_stocks = len(self._df_log_returns.columns)

        ax = (
            self._df_log_returns.melt()
            .reset_index()
            .rename(columns={0: "return"})
            .hist(
                column="return",
                by="Ticker",
                range=[self.data.min().min(), self.data.max().max()],
                bins=100,
                grid=False,
                figsize=(16, 16),
                layout=(n_stocks, 1),
                sharex=True,
                color="#86bf91",
                zorder=2,
                rwidth=0.9,
            )
        )

        for i, x in enumerate(ax):
            x.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                labelbottom=True,
                left=False,
                right=False,
                labelleft=True,
            )

            vals = x.get_yticks()
            for tick in vals:
                x.axhline(
                    y=tick, linestyle="dashed", alpha=0.4, color="#eeeeee", zorder=1
                )

            start_year = self.start.year if self.start else "Unknown"
            end_year = self.end.year if self.end else "Unknown"

            x.set_xlabel(
                f"Daily Return ({start_year}-{end_year})",
                labelpad=20,
                weight="bold",
                size=16,
            )

            x.set_title(f"{self.tickers[self.data.columns[i]]}", size=12)

            if i == n_stocks // 2:
                x.set_ylabel("Frequency", labelpad=50, weight="bold", size=12)

            x.tick_params(axis="x", rotation=0)

    def __getitem__(self, item: int | str, *args, **kwargs) -> np.ndarray:  # type: ignore[no-untyped-def, type-arg]  # noqa: ANN003, ANN002
        if self.data is None:
            msg = "Missing data for self.data"
            raise ValueError(msg)
        if isinstance(item, int):
            return self.data.iloc[:, item].to_numpy()
        if item in self.data.columns:
            return self.data.loc[:, item].to_numpy()
        return self.data.loc[:, self.tickers[item]].to_numpy()  # type: ignore[index]

    @cache  # noqa: B019
    def get_dividends(self) -> np.ndarray:  # type: ignore[type-arg]
        return np.array(
            [
                yfin.Ticker(ticker).dividends.iloc[-1] / 100
                for ticker in self.tickers.codes
            ]
        )

    def __repr__(self) -> str:
        s = "Underlyings:\n"
        for ticker in self.tickers:
            s += f"{ticker}\n"
        return s

    def __str__(self) -> str:
        return self.__repr__()
