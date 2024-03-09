from abc import ABC, abstractmethod

import netCDF4
import wandb


class PatchProcessor(ABC):
    @abstractmethod
    def process(self, path: str, target_split: str, netcdf_dataset: netCDF4.Dataset):
        pass

    @abstractmethod
    def log_to_artifact(self, artifact: wandb.Artifact):
        pass
