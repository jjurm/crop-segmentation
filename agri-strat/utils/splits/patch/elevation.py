import glob
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.merge
import rasterstats
import rioxarray
from rasterio import MemoryFile

from utils.splits.patch.stats_adder import PatchStatsAdder


class ElevationStats(PatchStatsAdder):
    def __init__(self, srtm_dataset_path: Path):
        files = glob.glob((srtm_dataset_path / "*.hgt.zip").as_posix())
        assert len(files) > 0, f"No SRTM30m files found in {srtm_dataset_path}."

        print("Elevation: loading SRTM30m data...")
        datasets = [rasterio.open(file) for file in files]
        print("Elevation: merging SRTM30m data...")
        mosaic, transform = rasterio.merge.merge(datasets)

        metadata = datasets[0].meta.copy()
        metadata.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
        })

        with MemoryFile() as memfile:
            with memfile.open(**metadata) as dataset:
                dataset.write(mosaic)
                del mosaic

            with memfile.open() as dataset:  # Reopen as DatasetReader
                self.array = rioxarray.open_rasterio(dataset)

    def _get_elevation(self, x: float, y: float) -> float:
        return self.array.sel(x=x, y=y, method="nearest").item()

    def process(self, geo_df: gpd.GeoDataFrame) -> pd.DataFrame:
        stats = rasterstats.zonal_stats(
            vectors=geo_df,
            raster=self.array.values.squeeze(axis=0),
            affine=self.array.rio.transform(),
            prefix="elevation_",
            stats="mean std median",
        )
        return pd.concat([geo_df, pd.DataFrame(stats)], axis=1)
