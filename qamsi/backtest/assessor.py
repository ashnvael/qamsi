from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import levene, ttest_ind


@dataclass
class StrategyStatistics:
    final_nav: float
    geom_avg_total_r: float
    geom_avg_xs_r: float
    std_xs_r: float
    min_xs_r: float
    max_xs_r: float
    skew: float
    kurtosis: float

    max_dd: float  # Maximum drawdown
    sharpe: float

    alpha_buy_hold: float
    tracking_error_buy_hold: float
    ir_buy_hold: float

    factor_loadings: dict[str, float]

    alpha_benchmark: float
    alpha_benchmark_pvalue: float
    tracking_error_benchmark: float
    ir_benchmark: float

    ttest_pval: float
    levene_pval: float

    timing_ability_coef: float
    timing_ability_pval: float

    def __str__(self) -> str:
        string = f"{self.__class__.__name__}:"
        for key, value in self.__dict__.items():
            val = f"{value:.6f}" if isinstance(value, float) else value
            string += f"\n* {key} = {val}"
        return string

    def __repr__(self) -> str:
        return str(self)


class Assessor:
    def __init__(
        self,
        rf_rate: pd.DataFrame,
        factors: pd.DataFrame,
        mkt_name: str = "spx",
        factor_annual: int | None = None,
    ) -> None:
        self.rf_rate = rf_rate
        self.factors = factors
        self.mkt_name = mkt_name
        self.factor_annual = factor_annual

    @staticmethod
    def _get_benchmark(
        strategy_excess: pd.Series, factors: pd.DataFrame | None
    ) -> tuple[Any, Any, Callable[[], Any]]:
        y = strategy_excess
        features = (
            factors
            if factors is not None
            else pd.DataFrame(index=strategy_excess.index)
        )

        features = sm.add_constant(features)

        lr = sm.OLS(y, features).fit()

        benchmark_excess_r = lr.predict(features) - lr.params.iloc[0]

        return benchmark_excess_r, lr.params.iloc[1:], lr.pvalues

    @staticmethod
    def _get_timing_ability(
        strategy_excess: pd.Series, factors: pd.DataFrame | None
    ) -> tuple[float, float]:
        y = strategy_excess

        features = (
            factors
            if factors is not None
            else pd.DataFrame(index=strategy_excess.index)
        )

        # Treynor-Mazuy procedure
        features_timing = pd.Series(np.maximum(features, 0))
        features = sm.add_constant(features)
        features = pd.concat((features, features_timing), axis=1)

        lr = sm.OLS(y, features).fit()

        return lr.params.iloc[-1].item(), lr.pvalues.iloc[-1].item()

    @staticmethod
    def _get_max_drawdowns(total_returns: pd.Series) -> float:
        total_nav = total_returns.add(1).cumprod()
        prev_peak = total_nav.cummax()
        return ((total_nav - prev_peak) / prev_peak).min()

    @staticmethod
    def _get_sharpe_ratio_pvalue(
        strategy_total_r: pd.Series, baseline_total_r: pd.Series
    ) -> float:
        raise NotImplementedError

    def _run(self, strategy_total: pd.Series) -> StrategyStatistics:
        if len(strategy_total.shape) > 1:
            strategy_total = strategy_total.iloc[:, 0]  # type: ignore  # noqa: PGH003

        if self.factor_annual is None:
            day_diff = strategy_total.index.diff().days
            factor_annual = round(np.nanmean(365 // day_diff))
        else:
            factor_annual = self.factor_annual
        n_periods = strategy_total.shape[0] / factor_annual

        buy_hold_total = self.factors[self.mkt_name].add(self.rf_rate, axis=0)

        final_nav = strategy_total.add(1).prod()
        final_buy_hold = buy_hold_total.add(1).prod()
        final_rf = self.rf_rate.add(1).prod()

        strat_mean = final_nav ** (1 / n_periods) - 1
        buy_hold_mean = final_buy_hold ** (1 / n_periods) - 1
        rf_mean = final_rf ** (1 / n_periods) - 1

        strat_excess_mean = strat_mean - rf_mean

        strategy_excess = strategy_total.sub(self.rf_rate, axis=0)
        buy_hold_excess = buy_hold_total.sub(self.rf_rate, axis=0)

        strat_std = strategy_excess.std() * np.sqrt(factor_annual)
        strat_min = strategy_excess.min()
        strat_max = strategy_excess.max()

        strat_skew = strategy_excess.skew()
        strat_kurtosis = strategy_excess.kurtosis()
        strategy_max_dd = self._get_max_drawdowns(strategy_excess)

        sr_strategy_total = strat_excess_mean / strat_std

        buy_hold_alpha = strat_mean - buy_hold_mean
        buy_hold_tracking_error = np.std(strategy_excess - buy_hold_excess) * np.sqrt(
            factor_annual
        )
        buy_hold_ir = buy_hold_alpha / buy_hold_tracking_error

        benchmark_excess, loadings, pvalues = self._get_benchmark(
            strategy_excess, self.factors
        )
        benchmark_total = benchmark_excess.add(self.rf_rate, axis=0)
        final_benchmark = benchmark_total.add(1).prod()
        benchmark_mean = final_benchmark ** (1 / n_periods) - 1

        benchmark_alpha = strat_mean - benchmark_mean
        benchmark_tracking_error = np.std(strategy_excess - benchmark_excess) * np.sqrt(
            factor_annual
        )
        benchmark_ir = benchmark_alpha / benchmark_tracking_error
        alpha_benchmark_pvalue = pvalues.iloc[0]  # type: ignore  # noqa: PGH003

        ttest_pval = ttest_ind(
            strategy_excess, buy_hold_excess, alternative="greater"
        ).pvalue
        levene_pval = levene(strategy_excess, buy_hold_excess).pvalue

        timing_ability_coef, timing_ability_pval = self._get_timing_ability(
            strategy_excess, buy_hold_excess
        )

        # TODO(@V): 3 autocorrelation lags

        return StrategyStatistics(
            final_nav=final_nav,
            geom_avg_total_r=strat_mean,
            geom_avg_xs_r=strat_excess_mean,
            std_xs_r=strat_std,
            min_xs_r=strat_min,
            max_xs_r=strat_max,
            skew=strat_skew,  # type: ignore  # noqa: PGH003
            kurtosis=strat_kurtosis,  # type: ignore  # noqa: PGH003
            sharpe=sr_strategy_total,
            max_dd=strategy_max_dd,
            alpha_buy_hold=buy_hold_alpha,
            tracking_error_buy_hold=buy_hold_tracking_error,
            ir_buy_hold=buy_hold_ir,
            alpha_benchmark=benchmark_alpha,
            alpha_benchmark_pvalue=alpha_benchmark_pvalue,
            tracking_error_benchmark=benchmark_tracking_error,
            ir_benchmark=benchmark_ir,
            factor_loadings=dict(zip(self.factors.columns, loadings, strict=False)),
            ttest_pval=ttest_pval,
            levene_pval=levene_pval,
            timing_ability_coef=timing_ability_coef,
            timing_ability_pval=timing_ability_pval,
        )

    def __call__(self, strategy_total: pd.Series) -> StrategyStatistics:
        return self._run(strategy_total)
