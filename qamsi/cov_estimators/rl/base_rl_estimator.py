from __future__ import annotations

from abc import abstractmethod
from enum import Enum

import pandas as pd

from qamsi.strategies.optimization_data import PredictionData, TrainingData
from qamsi.cov_estimators.shrinkage.rp_cov_estimator import RiskfolioCovEstimator


class ShrinkageType(Enum):
    LINEAR = "linear"
    QIS = "qis"


class BaseRLCovEstimator(RiskfolioCovEstimator):
    def __init__(self, shrinkage_type: str = "linear") -> None:
        self.shrinkage_type = ShrinkageType(shrinkage_type)

        # TODO(@V): Implement DNK QIS
        if self.shrinkage_type == ShrinkageType.QIS:
            msg = "QIS shrinkage not yet implemented"
            raise NotImplementedError(msg)
        elif self.shrinkage_type == ShrinkageType.LINEAR:
            super().__init__(
                estimator_type="shrunk",
                alpha=0.1, # Starting alpha, will be updated during fit
            )
        else:
            msg = f"Unknown shrinkage type: {self.shrinkage_type}"
            raise NotImplementedError(msg)

        self._fitted_cov = None
        self._seen_training_data = None

        self._predictions = []
        self.trained_with_features = False

    @abstractmethod
    def _fit_shrinkage(self, features: pd.DataFrame, shrinkage_target: pd.Series) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict_shrinkage(self, features: pd.DataFrame) -> float:
        raise NotImplementedError

    def _fit(self, training_data: TrainingData) -> None:
        self._seen_training_data = training_data
        feat = training_data.features
        last_pred = self.predictions
        target = training_data.targets["l_shrinkage"]

        first_feat, first_target = feat.first_valid_index(), target.first_valid_index()

        first_date = first_feat if first_feat >= first_target else first_target

        feat = feat.loc[first_date:]
        if not last_pred.empty:
            feat = pd.merge_asof(feat, last_pred, left_index=True, right_index=True, tolerance=pd.Timedelta("1D")).ffill()

            if feat["prediction"].isna().any():
                feat = feat.drop(["prediction"], axis=1)
                self.trained_with_features = False
            else:
                self.trained_with_features = True
        else:
            self.trained_with_features = False

        target = target.loc[first_date:]

        self._fit_shrinkage(features=feat, shrinkage_target=target)

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        feat = prediction_data.features
        if not self.predictions.empty and self.trained_with_features:
            feat["prediction"] = self.predictions.iloc[-1].item()
        pred_shrinkage = self._predict_shrinkage(feat)
        self.alpha = pred_shrinkage

        self._predictions.append([feat.index[-1], pred_shrinkage])

        super()._fit(training_data=self._seen_training_data)

        self._seen_training_data = None

        return self._fitted_cov

    @property
    def predictions(self) -> pd.DataFrame | None:
        if self._predictions is None:
            return None

        pred = pd.DataFrame(self._predictions, columns=["date", "prediction"])
        pred["date"] = pd.to_datetime(pred["date"])
        return pred.set_index("date")
