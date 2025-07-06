from __future__ import annotations

from dataclasses import dataclass

from config.us_experiment_config import ExperimentConfig as USExperimentConfig


@dataclass
class TopNExperimentConfig(USExperimentConfig):
    def __init__(self, topn: int = 20):
        self.topn = topn

        self.DF_FILENAME = f"top{self.topn}_data.csv"
        self.PRESENCE_MATRIX_FILENAME = f"top{self.topn}_presence_matrix.csv"
