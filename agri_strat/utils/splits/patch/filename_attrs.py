from pathlib import Path

import netCDF4
import pandas as pd

from agri_strat.utils.splits.patch.patch_processor import PatchProcessor


class PatchFilenameAttrs(PatchProcessor):
    def process(self, path: Path, row: pd.Series, netcdf_dataset: netCDF4.Dataset = None) -> pd.Series:
        year, tile, _, patch_x, patch_y = path.stem.split("_")
        row['year'] = int(year)
        row['tile'] = tile
        row['patch_x'] = int(patch_x)
        row['patch_y'] = int(patch_y)
        return row
