from __future__ import annotations

import numpy as np
import quantstats as qs

from qamsi.strategies.heuristics.equally_weighted import EWStrategy
from qamsi.cov_estimators.cov_estimators import CovEstimators
from qamsi.utils.data import read_csv
from run import Dataset

qs.extend_pandas()

REBAL_FREQ = "ME"
TOP_N = 30
DATASET = Dataset.TOPN_US
ESTIMATOR = CovEstimators.PRETRAINED.value(name="irl")
BASELINE = EWStrategy()

dataset = DATASET.value(topn=TOP_N)

strategy_name = DATASET.name + ESTIMATOR.__class__.__name__ + f"_rebal{REBAL_FREQ}"
strategy = read_csv(dataset.SAVE_PATH, strategy_name + ".csv")

strategy_excess_r = strategy["strategy_xs_r"]
rf = strategy["acc_rate"]

strategy_total_r = strategy["strategy_xs_r"].add(rf, axis=0)
bm = strategy["spx"].add(rf, axis=0).rename("spx")

qs.plots.snapshot(strategy_total_r, title="AIRL", show=True)

qs.plots.monthly_heatmap(
    strategy_total_r,
    benchmark=strategy["spx"].add(rf, axis=0),
    figsize=(12, 8),
    show=True,
    savefig=f"{dataset.SAVE_PATH}/monthly_{strategy_name}.pdf",
)

qs.plots.drawdowns_periods(
    strategy_total_r,
    title="AIRL",
    figsize=(12, 8),
    show=True,
    savefig=f"{dataset.SAVE_PATH}/drawdowns_{strategy_name}.pdf",
)

day_diff = strategy_total_r.index.diff().days
factor_annual = round(np.nanmean(365 // day_diff))
n_periods = strategy_total_r.shape[0] / factor_annual

final_rf = rf.add(1).prod()
rf_mean = final_rf ** (1 / n_periods) - 1

qs.reports.html(strategy_total_r, output=f"{dataset.SAVE_PATH}/report.html", benchmark=bm, rf=rf_mean, title=strategy_name)
