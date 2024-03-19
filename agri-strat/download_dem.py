import argparse
import os
from pathlib import Path

import pandas as pd
import geopandas as gpd

import wandb
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import urllib.request


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_artifact", type=str, required=True, help="Artifact of the type 'split'.")
    parser.add_argument("--dem_dir", type=Path, required=False, default=Path("dataset/dem/srtm30"),
                        help="Directory to save the SRTM30m data.")
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
    intersecting_boxes = intersecting_boxes.iloc[:2]

    # Download the SRTM30m data
    print("Downloading SRTM30m data...")
    bearer_token = os.getenv("EARTHDATA_TOKEN")
    base_url = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/"
    config["dem_dir"].mkdir(parents=True, exist_ok=True)
    with logging_redirect_tqdm():
        for i, row in tqdm.tqdm(intersecting_boxes.iterrows(), total=len(intersecting_boxes)):
            url = base_url + row["dataFile"]
            download_path = config["dem_dir"] / row["dataFile"]
            if not download_path.exists():
                opener = urllib.request.build_opener()
                opener.addheaders = [('Authorization', 'Bearer ' + bearer_token)]
                urllib.request.install_opener(opener)
                result = urllib.request.urlretrieve(url, download_path, data=None)
                print(f"Downloaded {result[0]}: {result[1]['Content-Length']} bytes.")
            else:
                print(f"Skipped existing {download_path}.")


if __name__ == "__main__":
    main()
