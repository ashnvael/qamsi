from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from qamsi.strategies.optimization_data import TrainingData, PredictionData
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator
from qamsi.features.covar import var_covar_from_corr_array


@dataclass
class SklearnParams:
    #TODO(@V): Add typing
    scaler = StandardScaler()
    random_state: int | None = None


class SklearnMlPredictor(BaseCovEstimator):
    def __init__(self, sklearn_model: BaseEstimator, sklearn_params: SklearnParams, resample_freq: str = "ME") -> None:
        super().__init__()

        self.sklearn_model = sklearn_model
        self.sklearn_params = sklearn_params
        self.resample_freq = resample_freq

        self._models = {}
        self._obs_means = {}

        self._obs_corr = None
        self._obs_var = None

        self.feat_scaler = sklearn_params.scaler

    def _fit(self, training_data: TrainingData) -> None:
        feat = training_data.features
        ret = training_data.simple_excess_returns

        self._obs_corr = ret.corr().to_numpy()
        self._obs_var = ret.resample("ME").var() * 252
        features = feat.resample("ME").mean()
        features = self.feat_scaler.fit_transform(features)
        for stock in self._available_assets:
            self._models[stock] = self.sklearn_model(**self.sklearn_params)
            self._models[stock].fit(features, self._obs_var[stock].ffill().fillna(0))

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        feat = prediction_data.features

        available_assets = self._available_assets
        variances = []
        features = self.feat_scaler.transform(feat)
        for stock in available_assets:
            variances.append(self._models[stock].predict(features).item())

        vols = np.eye(len(variances)) * np.sqrt(np.array(variances) / 20)
        cov = var_covar_from_corr_array(self._obs_corr, vols)
        covmat = pd.DataFrame(cov, index=available_assets, columns=available_assets)

        return covmat.astype(float)
