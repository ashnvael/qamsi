from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qamsi.config.us_experiment_config import USExperimentConfig as USExperimentConfig


@dataclass
class TopNExperimentConfig(USExperimentConfig):
    def __init__(self, topn: int = 20):
        super().__init__()

        self.TOPN = topn
        self.PREFIX = f"top{self.TOPN}_"

        self.PATH_TARGETS = Path(__file__).resolve().parents[2] / "data" / "targets"
