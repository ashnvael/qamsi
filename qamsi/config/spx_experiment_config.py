from __future__ import annotations

from dataclasses import dataclass

from qamsi.config.us_experiment_config import USExperimentConfig as USExperimentConfig


@dataclass
class SPXExperimentConfig(USExperimentConfig):
    def __init__(self):
        super().__init__()
        self.PREFIX = "spx_"
