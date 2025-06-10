from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qamsi.market_data.ticker import Ticker


class Tickers:
    def __init__(self, tickers: list[Ticker]) -> None:
        self.tickers_list = tickers

        self._initialize()

    def _initialize(self) -> None:
        self._ticker_dict = self._get_ticker_dict(self.tickers_list)
        self._ticker_inverse_dict = self._get_ticker_inverse_dict(self.tickers_list)

        self.names = list(self._ticker_inverse_dict.keys())
        self.codes = list(self._ticker_dict.keys())

    @staticmethod
    def _get_ticker_dict(tickers: list[Ticker]) -> dict[str, Ticker]:
        return {ticker.code: ticker for ticker in tickers}

    @staticmethod
    def _get_ticker_inverse_dict(tickers: list[Ticker]) -> dict[str, Ticker]:
        return {ticker.name: ticker for ticker in tickers}

    def __len__(self) -> int:
        return len(self.tickers_list)

    def __getitem__(self, item: int | str) -> Ticker | list[Ticker]:
        if isinstance(item, int):
            return self.tickers_list[item]
        if isinstance(item, str):
            if item in self._ticker_dict:
                return self.get(item)
            if item in self._ticker_inverse_dict:
                return self.get_inverse(item)

        msg = f"Item {item} is not a valid ticker or index"
        raise TypeError(msg)

    def __add__(self, other: Tickers) -> Tickers:
        return Tickers(self.tickers_list + other.tickers_list)

    def get(self, name: str) -> Ticker:
        return self._ticker_dict[name]

    def get_inverse(self, code: str) -> Ticker:
        return self._ticker_inverse_dict[code]

    def __iter__(self) -> Iterator[Ticker]:
        return iter(self.tickers_list)
