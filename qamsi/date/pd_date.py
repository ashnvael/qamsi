from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime as dt

import pandas as pd


def create_dates_df(
    start: str | dt.datetime,
    end: str | dt.datetime,
    freq: str = "D",
    name: str = "date",
) -> pd.DataFrame:
    return pd.date_range(start=start, end=end, freq=freq).to_frame(
        name=name, index=False
    )
