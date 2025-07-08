from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _mkt_cap_universe_builder_fn(
    crsp_data: pd.DataFrame,
    mkt_cap_filter_fn: Callable[
        [
            pd.Series,
        ],
        bool,
    ],
) -> pd.DataFrame:
    crsp_data["mktcap"] = crsp_data["shrout"] * 1_000 * crsp_data["prc"]
    mktcap_init = crsp_data.reset_index().pivot(
        index="date", columns="permno", values="mktcap"
    )
    mktcap = mktcap_init.resample("ME").last()
    mktcap = pd.concat([mktcap_init.iloc[:1], mktcap], axis=0)

    presence_matrix = mktcap.apply(lambda x: mkt_cap_filter_fn(x), axis=1).astype(float)
    presence_matrix[presence_matrix == 0] = np.nan

    return presence_matrix.dropna(axis=1, how="all")


def mkt_cap_topn_universe_builder_fn(
    crsp_data: pd.DataFrame, topn: int
) -> pd.DataFrame:
    return _mkt_cap_universe_builder_fn(
        crsp_data, lambda x: x >= x.nlargest(topn).min()
    )


def mkt_cap_quantile_universe_builder_fn(
    crsp_data: pd.DataFrame, quantile: float
) -> pd.DataFrame:
    return _mkt_cap_universe_builder_fn(crsp_data, lambda x: x >= x.quantile(quantile))
