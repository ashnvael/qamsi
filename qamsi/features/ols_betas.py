from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def prepare_data(
    risk_premia: pd.DataFrame, excess_r: pd.Series
) -> tuple[pd.Series, pd.DataFrame]:
    data = pd.merge_asof(
        risk_premia,
        excess_r,
        left_index=True,
        right_index=True,
        tolerance=pd.Timedelta("1D"),
    )
    data = data.dropna(axis=0, how="any")

    y = data[excess_r.name]
    x = data[risk_premia.columns]
    x = sm.add_constant(x)

    return y, x


def get_exposures(
    factors: pd.DataFrame, targets: pd.DataFrame, return_residuals: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    # TODO(@V): Speedup by vectorizing regressions as tensors

    betas = pd.DataFrame(index=targets.columns, columns=factors.columns)
    resids = pd.DataFrame(index=targets.index, columns=targets.columns)
    for stock in targets.columns:
        # Stocks should be passed as already excess returns
        xs_r = targets[stock]

        y, x = prepare_data(
            excess_r=xs_r,
            risk_premia=factors,
        )

        if len(x) == 0:
            betas.loc[stock] = np.nan
            resids[stock] = np.nan
        else:
            lr = sm.OLS(y, x)
            results = lr.fit()
            betas.loc[stock] = results.params.loc[betas.columns]
            resids[stock] = results.resid

    if return_residuals:
        return betas, resids

    return betas


def get_betas(market_index: pd.Series, targets: pd.DataFrame) -> pd.Series:
    # TODO(@V): Depreciate and use get_exposures()

    # Index should be passed as already excess returns
    erp = market_index

    betas = pd.DataFrame(index=targets.columns, columns=[market_index.name])
    for stock in targets.columns:
        # Stocks should be passed as already excess returns
        xs_r = targets[stock]

        y, x = prepare_data(
            excess_r=xs_r,
            risk_premia=erp.to_frame(),
        )

        if len(x) == 0:
            beta = 1.0
        else:
            lr = sm.OLS(y, x).fit()
            beta = lr.params.loc[erp.name].item()

        if np.isnan(beta):
            msg = f"Beta for {stock} is NaN."
            raise ValueError(msg)

        betas.loc[stock] = beta

    return betas.iloc[:, 0]


def get_window_betas(
    market_index: pd.Series, targets: pd.DataFrame, window_days: int | None
) -> pd.Series:
    first_date = (
        targets.index[-1] - pd.Timedelta(days=window_days)
        if window_days is not None
        else None
    )

    return get_betas(
        market_index=market_index.loc[first_date:],
        targets=targets.loc[first_date:],
    )
