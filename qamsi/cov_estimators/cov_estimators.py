from enum import Enum

from qamsi.cov_estimators.heuristic.hist_cov_estimator import HistoricalCovEstimator
from qamsi.cov_estimators.factor.factor_cov_estimator import FactorCovEstimator
from qamsi.cov_estimators.rl.behavioral_cloning.last_optimal_estimator import (
    LastOptimalCovEstimator,
)

from qamsi.cov_estimators.shrinkage.diag_hist_cov_estimator import (
    DiagHistoricalCovEstimator,
)
from qamsi.cov_estimators.shrinkage.pca_cov_estimator import PCACovEstimator
from qamsi.cov_estimators.shrinkage.qis import QISCovEstimator
from qamsi.cov_estimators.shrinkage.rp_cov_estimator import RiskfolioCovEstimator

from qamsi.cov_estimators.ml.predictors.sklearn_ml_predictor import SklearnMlPredictor
from qamsi.cov_estimators.ml.glasso_estimator import GLassoCovEstimator
from qamsi.cov_estimators.ml.glasso_tscv_estimator import GLassoTSCVCovEstimator

from qamsi.cov_estimators.rl.behavioral_cloning.ma_estimator import MACovEstimator
from qamsi.cov_estimators.rl.behavioral_cloning.dnk_estimator import DNKCovEstimator
from qamsi.cov_estimators.rl.behavioral_cloning.ols_estimator import OLSCovEstimator
from qamsi.cov_estimators.rl.behavioral_cloning.gpr_estimator import GPRCovEstimator
from qamsi.cov_estimators.rl.behavioral_cloning.pretrained_estimator import (
    PretrainedCovEstimator,
)
from qamsi.cov_estimators.rl.behavioral_cloning.uncert_ensemble_estimator import (
    UncertEnsembleCovEstimator,
)

from qamsi.cov_estimators.rl.behavioral_cloning.rf_estimator import (
    RandomForestCovEstimator,
)
from qamsi.cov_estimators.rl.behavioral_cloning.xgb_estimator import XGBCovEstimator
from qamsi.cov_estimators.rl.behavioral_cloning.dl_estimator import DLCovEstimator

from qamsi.cov_estimators.rl.behavioral_cloning.rf_xiu_estimator import (
    RandomForestXiuCovEstimator,
)


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

    MA = MACovEstimator
    LAST_OPTIMAL = LastOptimalCovEstimator
    DNK = DNKCovEstimator
    OLS = OLSCovEstimator
    GPR = GPRCovEstimator
    PRETRAINED = PretrainedCovEstimator
    UNCERT_ENSEMBLE = UncertEnsembleCovEstimator

    RF = RandomForestCovEstimator
    XGB = XGBCovEstimator
    DEEP_LEARNING = DLCovEstimator

    RF_XIU = RandomForestXiuCovEstimator
