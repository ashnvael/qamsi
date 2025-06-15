from enum import Enum

from qamsi.cov_estimators.heuristic.hist_cov_estimator import HistoricalCovEstimator
from qamsi.cov_estimators.factor.factor_cov_estimator import FactorCovEstimator

from qamsi.cov_estimators.shrinkage.diag_hist_cov_estimator import (
    DiagHistoricalCovEstimator,
)
from qamsi.cov_estimators.shrinkage.pca_cov_estimator import PCACovEstimator
from qamsi.cov_estimators.shrinkage.qis import QISCovEstimator
from qamsi.cov_estimators.shrinkage.rp_cov_estimator import RiskfolioCovEstimator

from qamsi.cov_estimators.ml.predictors.sklearn_ml_predictor import SklearnMlPredictor
from qamsi.cov_estimators.ml.glasso_estimator import GLassoCovEstimator
from qamsi.cov_estimators.ml.glasso_tscv_estimator import GLassoTSCVCovEstimator

from qamsi.cov_estimators.rl.ma_linear_estimator import MALinearCovEstimator
from qamsi.cov_estimators.rl.dnk_linear_estimator import DNKLinearCovEstimator
from qamsi.cov_estimators.rl.gpr_linear_estimator import GPRLinearCovEstimator
from qamsi.cov_estimators.rl.gp_ucb_estimator import GPUCBCovEstimator
from qamsi.cov_estimators.rl.uncert_ensemble_estimator import UncertEnsembleCovEstimator

from qamsi.cov_estimators.rl.rf_estimator import RandomForestCovEstimator


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

    MA = MALinearCovEstimator
    DNK = DNKLinearCovEstimator
    GPR = GPRLinearCovEstimator
    GP_UCB = GPUCBCovEstimator
    UNCERT_ENSEMBLE = UncertEnsembleCovEstimator

    RF = RandomForestCovEstimator
