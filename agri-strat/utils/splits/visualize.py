from pathlib import Path

import geopandas as gpd
import netCDF4
import pandas as pd
import shapely
import wandb
from tqdm import tqdm


def create_split_visualization(df, data_path):
    """
    Create a GPKG visualization of the patches in each split.
    :param df: A dataframe with columns "path" and "target"
    :param data_path: The path to the netCDF files
    :return: The path to the created GPKG file
    """
    geo_dfs = []
    print("Creating split visualization...")
    with tqdm(total=len(df)) as pbar:
        for target, split in df.groupby("target"):
            for i, row in split.iterrows():
                with netCDF4.Dataset(data_path / row["path"]) as dataset:
                    crs = dataset["labels"].variables["labels"].crs.removeprefix("+init=")
                    transform = dataset["labels"].variables["labels"].transform
                    shape = dataset["labels"].variables["labels"].shape

                # Create a shapely polygon
                polygon = shapely.geometry.Polygon([
                    (transform[2], transform[5]),
                    (transform[2] + transform[0] * shape[0], transform[5]),
                    (transform[2] + transform[0] * shape[0], transform[5] + transform[4] * shape[1]),
                    (transform[2], transform[5] + transform[4] * shape[1]),
                ])

                geo_df = gpd.GeoDataFrame(geometry=[polygon], crs=crs)
                geo_df["split"] = target
                geo_df["year"] = int((data_path / row["path"]).name.split("_")[0])
                geo_df.to_crs("EPSG:4326", inplace=True)
                geo_dfs.append(geo_df)

                pbar.update(1)
    gpkg_path = Path(wandb.run.dir) / "splits.gpkg"
    pd.concat(geo_dfs).to_file(gpkg_path)
    return gpkg_path
