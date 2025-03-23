from enum import Enum

from qamsi.cov_estimators.hist_cov_estimator import HistoricalCovEstimator
from qamsi.cov_estimators.factor_cov_estimator import FactorCovEstimator
from qamsi.cov_estimators.ml.gbm_predictor import GBMPredictor


class CovEstimators(Enum):
    HISTORICAL = HistoricalCovEstimator
    STATIC_FACTOR_MODEL = FactorCovEstimator
    GBM_PREDICTOR = GBMPredictor
