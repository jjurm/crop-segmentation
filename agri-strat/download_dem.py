import argparse
import os
import urllib.request
from pathlib import Path

import geopandas as gpd
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_artifact", type=str, required=True, help="Artifact of the type 'split'.")
    parser.add_argument("--dem_path", type=Path, default=None, required=False,
                        help="Directory to save the SRTM30m data. Default: $DEM_PATH or 'dataset/dem/srtm30'.")
    return parser.parse_args()


def main():
    config = vars(parse_args())
    api = wandb.Api()

    # Read the SRTM30m shapes
    srtm30_boxes = gpd.read_file("dataset/shapes/srtm30m_bounding_boxes.json")

    # Read the patch polygons
    print("Reading splits artifact...")
    splits_artifact = api.artifact(config["split_artifact"], type='split')
    splits_polygons = splits_artifact.get_entry("splits_polygons.gpkg").download()
    splits_gdf = gpd.read_file(splits_polygons)

    # Get all shapes that have an intersection with the splits
    print("Finding intersecting SRTM30m shapes...")
    srtm30_boxes["intersects"] = srtm30_boxes.intersects(splits_gdf.unary_union)
    intersecting_boxes = srtm30_boxes[srtm30_boxes["intersects"]]
    intersecting_boxes = intersecting_boxes

    # Download the SRTM30m data
    print("Downloading SRTM30m data...")
    bearer_token = os.getenv("EARTHDATA_TOKEN")
    if not bearer_token:
        raise ValueError("Please set the EARTHDATA_TOKEN environment variable.")
    base_url = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/"
    dem_path = config["dem_path"] or Path(os.getenv("DEM_PATH", "dataset/dem/srtm30"))
    dem_path.mkdir(parents=True, exist_ok=True)
    # noinspection PyArgumentList
    with logging_redirect_tqdm():
        for i, row in tqdm.tqdm(intersecting_boxes.iterrows(), total=len(intersecting_boxes)):
            url = base_url + row["dataFile"]
            download_path = dem_path / row["dataFile"]
            if not download_path.exists():
                opener = urllib.request.build_opener()
                opener.addheaders = [('Authorization', 'Bearer ' + bearer_token)]
                urllib.request.install_opener(opener)
                result = urllib.request.urlretrieve(url, download_path, data=None)
                print(f"Downloaded {result[0]}: {(result[1]['Content-Length'] // 1024) / 1024}MB.")
            else:
                print(f"Skipped existing {download_path}.")


if __name__ == "__main__":
    main()
