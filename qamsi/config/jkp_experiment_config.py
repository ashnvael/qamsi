from __future__ import annotations

from dataclasses import dataclass

from qamsi.config.us_experiment_config import USExperimentConfig as USExperimentConfig


@dataclass
class JKPExperimentConfig(USExperimentConfig):
    def __init__(self):
        super().__init__()
        self.MCAP_SELECTION_QUANTILE = 0.8
        self.PREFIX = "jkp_"
