from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, asdict
from typing import Type

import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from qamsi.cov_estimators.rl.dl.models.mlp import MLPPredictor


class AvailableModels(Enum):
    MLP = "mlp"


@dataclass
class ModelConfig:
    lr: float = 1e-3
    hidden_size: int = 32  # Matching BasicRewardNet for AIRL / GAIL
    n_layers: int = 2  # Matching BasicRewardNet for AIRL / GAIL
    dropout: float = 0.0

    n_epochs: int = 10
    n_features: int | None = None
    n_unique_features: int | None = None

    optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD
    scheduler: Type[torch.optim.lr_scheduler._LRScheduler] = (
        torch.optim.lr_scheduler.CosineAnnealingLR
    )

    weights_decay: float = 0.0

    batch_size: int = 64

    loss: nn.Module = nn.MSELoss()

    clip_grad_norm: float | None = None

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


class DeepLearningModel:
    def __init__(
        self,
        n_features: int,
        model_cls: Type[nn.Module] | str = "mlp",
        model_config: ModelConfig = ModelConfig(),
        verbose: bool = False,
    ):
        self.model_config = model_config
        self.model_config.n_features = n_features

        if isinstance(model_cls, str):
            model_cls = AvailableModels(model_cls).value

            if model_cls == "mlp":
                self.model = MLPPredictor(**model_config.dict())
        else:
            self.model = model_cls(**model_config.dict())

        self.model = self.model.to(model_config.device)
        self.device = model_config.device
        self.verbose = verbose

        self.optimizer = model_config.optimizer(
            self.model.parameters(),
            lr=model_config.lr,
            weight_decay=model_config.weights_decay,
        )
        self.scheduler = model_config.scheduler(
            self.optimizer, T_max=model_config.n_epochs
        )
        self.criterion = model_config.loss

    def fit(self, X: pd.DataFrame, y: pd.Series):
        features = torch.Tensor(X.to_numpy())
        targets = torch.Tensor(y.to_numpy())

        train_set = TensorDataset(features, targets)
        train_loader = DataLoader(
            train_set,
            batch_size=self.model_config.batch_size,
            shuffle=False,  # time series training
            pin_memory=False,  # due to mps training
            drop_last=False,
        )

        self._train_model(train_loader)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        feat_torch = torch.Tensor(X.to_numpy()).to(self.device)
        model_output = self.model(feat_torch).detach().cpu().numpy().flatten()
        return pd.Series(model_output, index=X.index, name="dl_prediction")

    def _train_model(self, train_loader: DataLoader) -> None:
        iter = range(self.model_config.n_epochs)
        if self.verbose:
            iter = tqdm(iter)

        for _ in (pbar := iter):
            train_loss = 0.0
            self.model.train()
            for features, labels in train_loader:
                self.optimizer.zero_grad()

                features = features.to(self.device)
                labels = labels.to(self.device)

                pred = self.model(features)

                loss = self.criterion(pred, labels)

                loss.backward()
                self.optimizer.step()

                if self.model_config.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.model_config.clip_grad_norm,
                    )

                train_loss += loss.item()

            train_loss /= len(train_loader.dataset)
            self.scheduler.step()
            if self.verbose:
                pbar.set_description(f"Loss: {train_loss:.4f}")

    def set_n_features(self, n_features: int) -> None:
        self.model_config.n_features = n_features
