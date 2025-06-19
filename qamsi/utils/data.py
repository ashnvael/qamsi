from pathlib import Path

import pandas as pd


def read_csv(path: str, filename: str, date_column: str = "date") -> pd.DataFrame:
    data = pd.read_csv(Path(path) / filename)
    data[date_column] = pd.to_datetime(data[date_column])
    return data.set_index(date_column).sort_index()
