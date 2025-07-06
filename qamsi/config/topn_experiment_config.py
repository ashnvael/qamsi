from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qamsi.config.us_experiment_config import ExperimentConfig as USExperimentConfig


@dataclass
class TopNExperimentConfig(USExperimentConfig):
    def __init__(self, topn: int = 20):
        self.topn = topn

        self.DF_FILENAME = f"top{self.topn}_" + self.DF_FILENAME
        self.PRESENCE_MATRIX_FILENAME = f"top{self.topn}_" + self.PRESENCE_MATRIX_FILENAME
        self.TRADE_DATASET_TMP_FILENAME = f"top{self.topn}_" + self.TRADE_DATASET_TMP_FILENAME

        self.PATH_TARGETS = Path(__file__).resolve().parents[2] / "data" / "targets"
