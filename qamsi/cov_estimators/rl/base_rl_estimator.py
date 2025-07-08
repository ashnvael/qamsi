from __future__ import annotations

from abc import abstractmethod
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from qamsi.strategies.optimization_data import PredictionData, TrainingData
from qamsi.cov_estimators.base_cov_estimator import BaseCovEstimator
from qamsi.cov_estimators.shrinkage.rp_cov_estimator import RiskfolioCovEstimator
from qamsi.cov_estimators.shrinkage.qis import QISCovEstimator


class ShrinkageType(Enum):
    LINEAR = "linear"
    QIS = "qis"


class BaseRLCovEstimator(BaseCovEstimator):
    def __init__(
        self,
        shrinkage_type: str = "linear",
        window_size: int | None = None,
        refit: bool = True,
    ) -> None:
        super().__init__()

        self.shrinkage_type = ShrinkageType(shrinkage_type)
        self.window_size = window_size

        self.refit = refit
        self._trained = False

        self.feat_scaler = StandardScaler()

        if self.shrinkage_type == ShrinkageType.QIS:
            self.shrinkage = QISCovEstimator(
                shrinkage=1.0,  # Starting shrinkage, will be updated during fit
            )
        elif self.shrinkage_type == ShrinkageType.LINEAR:
            self.shrinkage = RiskfolioCovEstimator(
                estimator_type="shrunk",
                alpha=0.1,  # Starting alpha, will be updated during fit
            )
        else:
            msg = f"Unknown shrinkage type: {self.shrinkage_type}"
            raise NotImplementedError(msg)

        self._seen_training_data = None

        self._predictions = []
        self.trained_with_features = False

    @abstractmethod
    def _fit_shrinkage(
        self, features: pd.DataFrame, shrinkage_target: pd.Series
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict_shrinkage(self, features: pd.DataFrame) -> float:
        raise NotImplementedError

    def _fit(self, training_data: TrainingData) -> None:
        self._seen_training_data = training_data

        if self._trained and not self.refit:
            return

        feat = training_data.features

        if self.shrinkage_type == ShrinkageType.LINEAR:
            target = training_data.targets["target"]
        elif self.shrinkage_type == ShrinkageType.QIS:
            target = training_data.targets["qis_shrinkage"]
        else:
            msg = f"Unknown shrinkage type: {self.shrinkage_type}"
            raise NotImplementedError(msg)

        start_date = (
            target.index[-1] - pd.Timedelta(days=self.window_size)
            if self.window_size is not None
            else target.index[0]
        )
        feat = feat.loc[start_date:]
        target = target.loc[start_date:]

        last_pred = self.predictions

        first_feat_date = feat.dropna(axis=0, how="any").first_valid_index()
        last_feat_date = feat.dropna(axis=0, how="any").last_valid_index()

        first_target_date = target.dropna(axis=0, how="any").first_valid_index()
        last_target_date = target.dropna(axis=0, how="any").last_valid_index()

        first_date = (
            first_feat_date
            if first_feat_date >= first_target_date
            else first_target_date
        )
        last_date = (
            last_feat_date if last_target_date >= last_feat_date else last_target_date
        )

        feat = feat.loc[first_date:last_date]
        if not last_pred.empty:
            feat = pd.merge_asof(
                feat,
                last_pred,
                left_index=True,
                right_index=True,
                tolerance=pd.Timedelta("1D"),
            ).ffill()

            if feat["prediction"].isna().any():
                feat = feat.drop(["prediction"], axis=1)
                self.trained_with_features = False
            else:
                self.trained_with_features = True
        else:
            self.trained_with_features = False

        target = target.loc[first_date:last_date]

        feat_transf = self.feat_scaler.fit_transform(feat)
        feat = pd.DataFrame(feat_transf, index=feat.index, columns=feat.columns)

        self.shrinkage.available_assets = self.available_assets
        self._fit_shrinkage(features=feat, shrinkage_target=target)

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        feat = prediction_data.features
        if not self.predictions.empty and self.trained_with_features:
            feat["prediction"] = self.predictions.iloc[-1].item()
        feat_transformed = self.feat_scaler.transform(feat)
        feat = pd.DataFrame(feat_transformed, index=feat.index, columns=feat.columns)

        pred_shrinkage = self._predict_shrinkage(feat)

        pred_shrinkage = (
            np.clip(pred_shrinkage, 0, 1)
            if self.shrinkage_type == ShrinkageType.LINEAR
            else pred_shrinkage
        )

        if self.shrinkage_type == ShrinkageType.LINEAR:
            self.shrinkage.alpha = pred_shrinkage
        elif self.shrinkage_type == ShrinkageType.QIS:
            self.shrinkage.shrinkage = pred_shrinkage
        else:
            msg = f"Unknown shrinkage type: {self.shrinkage_type}"
            raise NotImplementedError(msg)

        self._predictions.append([feat.index[-1], pred_shrinkage])

        self.shrinkage.fit(training_data=self._seen_training_data)

        self._seen_training_data = None

        return self.shrinkage.predict(prediction_data=prediction_data)

    @property
    def predictions(self) -> pd.DataFrame | None:
        if self._predictions is None:
            return None

        pred = pd.DataFrame(self._predictions, columns=["date", "prediction"])
        pred["date"] = pd.to_datetime(pred["date"])
        return pred.set_index("date")
