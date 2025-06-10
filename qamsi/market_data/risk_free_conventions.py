from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def accrue_risk_free_rate(rf_rate: pd.Series, calendar_days: int | None = None) -> pd.Series:
    days_diff = rf_rate.index.diff().days
    return rf_rate * days_diff if calendar_days is None else rf_rate * days_diff / calendar_days
