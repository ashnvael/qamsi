from enum import Enum

from qamsi.cov_estimators.heuristic.hist_cov_estimator import HistoricalCovEstimator
from qamsi.cov_estimators.factor.factor_cov_estimator import FactorCovEstimator
from qamsi.cov_estimators.ml.predictors.sklearn_ml_predictor import SklearnMlPredictor
from qamsi.cov_estimators.shrinkage.diag_hist_cov_estimator import (
    DiagHistoricalCovEstimator,
)
from qamsi.cov_estimators.shrinkage.pca_cov_estimator import PCACovEstimator
from qamsi.cov_estimators.shrinkage.qis import QISCovEstimator
from qamsi.cov_estimators.shrinkage.rp_cov_estimator import RiskfolioCovEstimator
from qamsi.cov_estimators.ml.glasso_estimator import GLassoCovEstimator
from qamsi.cov_estimators.ml.glasso_tscv_estimator import GLassoTSCVCovEstimator
from qamsi.cov_estimators.rl.dnk_linear_estimator import DNKLinearCovEstimator


class CovEstimators(Enum):
    HISTORICAL = HistoricalCovEstimator
    STATIC_FACTOR_MODEL = FactorCovEstimator
    DIAG_HISTORICAL = DiagHistoricalCovEstimator

    RISKFOLIO = RiskfolioCovEstimator

    SHRINKAGE_PCA = PCACovEstimator
    QIS = QISCovEstimator

    SKLEARN_ML_PREDICTOR = SklearnMlPredictor

    GLASSO = GLassoCovEstimator
    GLASSO_TSCV = GLassoTSCVCovEstimator

    DNK = DNKLinearCovEstimator
