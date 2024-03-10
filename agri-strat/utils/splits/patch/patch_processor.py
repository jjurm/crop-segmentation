from abc import ABC, abstractmethod
from pathlib import Path

import netCDF4
import wandb


class PatchProcessor(ABC):
    @abstractmethod
    def process(self, path: Path, target_split: str, netcdf_dataset: netCDF4.Dataset):
        pass

    @abstractmethod
    def log_to_artifact(self, artifact: wandb.Artifact):
        pass
