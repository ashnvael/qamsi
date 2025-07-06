from pathlib import Path

import pandas as pd


def read_csv(
    path: str | Path, filename: str, date_column: str = "date"
) -> pd.DataFrame:
    path = Path(path) if isinstance(path, str) else path
    data = pd.read_csv(path / filename)
    data[date_column] = pd.to_datetime(data[date_column])
    return data.set_index(date_column).sort_index()
