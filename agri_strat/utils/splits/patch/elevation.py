import glob
from pathlib import Path

import netCDF4
import pandas as pd
import rasterio
import rasterio.merge
import rasterstats
import rioxarray
from rasterio import MemoryFile

from utils.splits.patch.patch_processor import PatchProcessor


class PatchElevationStats(PatchProcessor):
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

    def process(self, path: Path, row: pd.Series, netcdf_dataset: netCDF4.Dataset = None) -> pd.Series:
        stats = rasterstats.zonal_stats(
            vectors=row["geometry"],
            raster=self.array.values.squeeze(axis=0),
            nodata=self.array.rio.nodata,
            affine=self.array.rio.transform(),
            prefix="elevation_",
            stats="mean std median",
        )
        return pd.concat([row, pd.Series(stats[0])])
