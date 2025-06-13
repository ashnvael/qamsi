from pathlib import Path

import pandas as pd


def read_csv(path: str, filename: str) -> pd.DataFrame:
    data = pd.read_csv(Path(path) / filename)
    data["date"] = pd.to_datetime(data["date"])
    return data.set_index("date")
