from enum import Enum

from qamsi.cov_estimators.hist_cov_estimator import HistoricalCovEstimator


class CovEstimators(Enum):
    HISTORICAL = HistoricalCovEstimator
