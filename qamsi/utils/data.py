from pathlib import Path

import pandas as pd


def read_csv(
    path: str | Path,
    filename: str,
    date_column: str = "date",
    rename_column: bool = False,
) -> pd.DataFrame:
    path = Path(path) if isinstance(path, str) else path
    data = pd.read_csv(path / filename)
    data[date_column] = pd.to_datetime(data[date_column])
    if rename_column:
        data = data.rename(columns={date_column: "date"})
        date_column = "date"
    return data.set_index(date_column).sort_index()
