from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import netCDF4
import pandas as pd
import wandb


class PatchProcessor(ABC):
    @abstractmethod
    def process(self, path: Path, row: pd.Series, netcdf_dataset: netCDF4.Dataset = None) -> pd.Series:
        pass

    def log_to_artifact(self, split_df: pd.DataFrame, artifact: wandb.Artifact):
        pass


class PatchApplyFn:
    def __init__(
            self,
            patch_processors: list[PatchProcessor],
            with_netcdf_file: bool = False,
            netcdf_path: Path = None
    ):
        self.patch_processors = patch_processors
        self.with_netcdf_file = with_netcdf_file
        self.netcdf_path = netcdf_path

    def _get_patch_dataset(self, relative_patch_path):
        return netCDF4.Dataset(self.netcdf_path / relative_patch_path)

    def __call__(self, row: pd.Series):
        path = cast(Path, row.name)
        with self._get_patch_dataset(path) if self.with_netcdf_file else nullcontext() as dataset:
            for patch_processor in self.patch_processors:
                row = patch_processor.process(path, row, netcdf_dataset=dataset)
        return row
