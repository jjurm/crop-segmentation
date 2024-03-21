from pathlib import Path

import geopandas as gpd
import netCDF4
import pandas as pd
import shapely
import wandb

from utils.splits.patch.patch_processor import PatchProcessor
from utils.splits.patch.stats_adder import PatchStatsAdder


class PatchVisualizer(PatchProcessor):
    """
    Create a GPKG visualization of the patches in each split.
    """

    def __init__(self, stats_adders: list[PatchStatsAdder]):
        self.stats_adders = stats_adders
        self.geo_dfs = []

    def process(self, path: Path, target_split: str, netcdf_dataset: netCDF4.Dataset):
        crs = netcdf_dataset["labels"].variables["labels"].crs.removeprefix("+init=")
        transform = netcdf_dataset["labels"].variables["labels"].transform
        shape = netcdf_dataset["labels"].variables["labels"].shape

        # Create a shapely polygon
        polygon = shapely.geometry.Polygon([
            (transform[2], transform[5]),
            (transform[2] + transform[0] * shape[0], transform[5]),
            (transform[2] + transform[0] * shape[0], transform[5] + transform[4] * shape[1]),
            (transform[2], transform[5] + transform[4] * shape[1]),
        ])

        geo_df = gpd.GeoDataFrame(geometry=[polygon], crs=crs)
        geo_df["split"] = target_split
        year, tile, _, patch_x, patch_y = path.stem.split("_")
        geo_df["path"] = path.as_posix()
        geo_df["year"] = int(year)
        geo_df["tile"] = tile
        geo_df["patch_x"] = int(patch_x)
        geo_df["patch_y"] = int(patch_y)
        geo_df.to_crs("EPSG:4326", inplace=True)

        for stats_adder in self.stats_adders:
            geo_df = stats_adder.process(geo_df)

        self.geo_dfs.append(geo_df)

    def _create_gpkg(self):
        gpkg_path = Path(wandb.run.dir) / "splits_polygons.gpkg"
        pd.concat(self.geo_dfs).to_file(gpkg_path)
        return gpkg_path

    def log_to_artifact(self, artifact: wandb.Artifact):
        gpkg_path = self._create_gpkg()
        artifact.add_file(gpkg_path.as_posix())
