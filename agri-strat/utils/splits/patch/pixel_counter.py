import netCDF4
import numpy as np
import wandb
import xarray as xr
import pandas as pd

from mappings.encodings_en import CROP_ENCODING_REVERSE
from utils.constants import LABEL_DTYPE
from utils.splits.patch.patch_processor import PatchProcessor


class PixelCounter(PatchProcessor):
    def __init__(self):
        self.counts = {}

    def process(self, path: str, target_split: str, netcdf_dataset: netCDF4.Dataset):
        labels = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf_dataset['labels']))['labels'].values \
            .astype(LABEL_DTYPE)

        # Compute pixel counts for each class
        if target_split not in self.counts:
            self.counts[target_split] = {}
        for c in map(int, np.unique(labels)):
            self.counts[target_split][c] = self.counts[target_split].get(c, 0) + int(np.sum(labels == c))

    def _get_count_dataframe(self, target):
        df = pd.DataFrame.from_records([
            (c, CROP_ENCODING_REVERSE.get(c, ""), count)
            for c, count in self.counts[target].items()
        ], columns=['class', 'class_name', 'count'])
        df = df.sort_values(by='class')
        return df

    def log_to_artifact(self, artifact: wandb.Artifact):
        for target in self.counts.keys():
            artifact.add(wandb.Table(
                dataframe=self._get_count_dataframe(target)
            ), name=f"pixel_counts_{target}")
