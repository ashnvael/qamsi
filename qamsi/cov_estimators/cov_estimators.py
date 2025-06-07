from enum import Enum

from qamsi.cov_estimators.hist_cov_estimator import HistoricalCovEstimator
from qamsi.cov_estimators.factor_cov_estimator import FactorCovEstimator
from qamsi.cov_estimators.ml.gbm_predictor import GBMPredictor
from qamsi.cov_estimators.ml.factor_predictor import FactorPredictor
from qamsi.cov_estimators.diag_hist_cov_estimator import DiagHistoricalCovEstimator
from qamsi.cov_estimators.shrinkage.pca_cov_estimator import PCACovEstimator
from qamsi.cov_estimators.shrinkage.qis import QISCovEstimator
from qamsi.cov_estimators.factor_cov_estimator2 import FactorCovEstimator as FactorCovEstimator2


class CovEstimators(Enum):
    HISTORICAL = HistoricalCovEstimator
    STATIC_FACTOR_MODEL = FactorCovEstimator
    DIAG_HISTORICAL = DiagHistoricalCovEstimator

    SHRINKAGE_PCA = PCACovEstimator
    QIS = QISCovEstimator

    FACTOR_PREDICTOR = FactorPredictor
    GBM_PREDICTOR = GBMPredictor

    NIKOLAI_COV = FactorCovEstimator2
