from pathlib import Path

import geopandas as gpd
import netCDF4
import pandas as pd
import shapely
import shapely.ops
import wandb
from pyproj import Transformer

from utils.splits.patch.patch_processor import PatchProcessor

TARGET_CRS = "EPSG:4326"


class PatchGeometry(PatchProcessor):
    """
    Create a GPKG visualization of the patches in each split.
    """

    def process(self, path: Path, row: pd.Series, netcdf_dataset: netCDF4.Dataset = None) -> pd.Series:
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

        transformer = Transformer.from_crs(crs, TARGET_CRS, always_xy=True)
        polygon_transformed = shapely.ops.transform(transformer.transform, polygon)
        row["geometry"] = polygon_transformed
        return row

    def log_to_artifact(self, split_df: pd.DataFrame, artifact: wandb.Artifact):
        path = (Path(wandb.run.dir) / "splits_polygons.gpkg").as_posix()
        gdf = gpd.GeoDataFrame(split_df, crs=TARGET_CRS)
        gdf.index = gdf.index.map(str)
        gdf.to_file(path)
        artifact.add_file(path)
