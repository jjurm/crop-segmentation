from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import wandb
import xarray as xr

from agri_strat.mappings.encodings_en import CROP_ENCODING_REVERSE
from agri_strat.utils.constants import LABEL_DTYPE
from agri_strat.utils.splits.patch.patch_processor import PatchProcessor


class ClassPixelCounts(PatchProcessor):
    def __init__(self):
        self.classes = set()

    def process(self, path: Path, row: pd.Series, netcdf_dataset: netCDF4.Dataset = None) -> pd.Series:
        labels = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf_dataset['labels']))['labels'].values \
            .astype(LABEL_DTYPE)

        classes = list(map(int, np.unique(labels)))
        self.classes = self.classes.union(classes)

        pixel_counts = {
            c: int(np.sum(labels == c))
            for c in classes
        }
        row["pixel_counts"] = pixel_counts
        return row


class ClassPixelCountsPerSplit(PatchProcessor):
    def __init__(self, classes: set[int]):
        self.classes = sorted(classes)
        self.counts = {}

    def process(self, path: Path, row: pd.Series, netcdf_dataset: netCDF4.Dataset = None) -> pd.Series:
        target_split = row["target"]
        pixel_counts = row["pixel_counts"]
        row.drop("pixel_counts", inplace=True)

        if target_split not in self.counts:
            self.counts[target_split] = {c: 0 for c in self.classes}

        for c in self.classes:
            count = pixel_counts.get(c, 0)
            self.counts[target_split][c] += count
            row["pixel_count_" + str(c)] = count
        row["pixel_count_max"] = max(v for c, v in pixel_counts.items() if c != 0)
        return row

    def _get_count_dataframe(self, target):
        df = pd.DataFrame.from_records([
            (c, CROP_ENCODING_REVERSE.get(c, ""), count)
            for c, count in self.counts[target].items()
        ], columns=['class', 'class_name', 'count'])
        df = df.sort_values(by='class')
        return df

    def log_to_artifact(self, split_df: pd.DataFrame, artifact: wandb.Artifact):
        for target in self.counts.keys():
            artifact.add(wandb.Table(
                dataframe=self._get_count_dataframe(target)
            ), name=f"pixel_counts_{target}")
