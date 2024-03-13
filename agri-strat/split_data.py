#!/usr/bin/env python
"""This file reads a yaml file from a wandb artifact describing the splits over patches and splits the data into
train, val and test sets, writing the results of each split to a separate txt file.

Information about each patch (year, tile, patch_x, patch_y) is taken from the file name, e.g. 2019_31TCJ_patch_20_29.nc
"""

import argparse
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from tqdm import tqdm

import wandb
from utils.splits.coco import split_by_coco_files
from utils.splits.patch.patch_processor import PatchProcessor
from utils.splits.patch.pixel_counter import PixelCounter
from utils.splits.patch.visualize import PatchVisualizer
from utils.splits.split_rules import split_by_split_rules


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_rules_artifact', type=str, required=False,
                        help='Wandb artifact of the \'split_rules\' type.')
    parser.add_argument('--coco_path_prefix', type=str, required=False, )
    parser.add_argument('--netcdf_path', type=str, default='dataset/netcdf', required=False,
                        help='Path to the netCDF files. Default "dataset/netcdf".')
    parser.add_argument('--seed', type=int, default=0, required=False,
                        help='Seed for random splits and shuffling. Default: 0.')
    parser.add_argument('--shuffle', action='store_true', default=False, required=False,
                        help='Shuffle the order of patches written out. Default: False.')
    parser.add_argument('--artifact_name_prefix', type=str, default=None, required=False,
                        help='Prefix for the name of the artifact that will be logged.')
    parser.add_argument('--limit_patches', type=int, default=None, required=False,
                        help='Limit the number of patches to process, for debugging. Default: None.')
    return parser.parse_args()


def create_artifact(
        config,
        split_rules_name: str | None,
        split_df: pd.DataFrame,
        patch_processors: list[PatchProcessor],
):
    print("Creating artifact...")
    artifact_name_prefix = config["artifact_name_prefix"] or split_rules_name
    artifact = wandb.Artifact(
        name=f"{artifact_name_prefix}_split",
        type="split",
        metadata={
            "split_rules_name": split_rules_name,
        },
    )

    # Write a txt file with the relative paths of the patches for each split
    splits_dir = Path(wandb.run.dir) / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    for target, target_df in split_df.groupby('target'):
        target_file_path = splits_dir / f"{target}.txt"
        target_df.drop(columns='target').to_csv(target_file_path, index=False, header=False)
        print(f"Split {target} written to {target_file_path}.")
    artifact.add_dir(splits_dir.as_posix(), name="splits")

    # Log number of patches per split
    patch_counts = split_df['target'].value_counts()
    print(patch_counts.to_string(header=False))
    artifact.add(wandb.Table(dataframe=patch_counts.to_frame().reset_index()), name="patch_counts")

    for patch_processor in patch_processors:
        patch_processor.log_to_artifact(artifact)

    wandb.run.log_artifact(artifact)


def main():
    with wandb.init(
            project='agri-strat',
            job_type='generate-split',
            config=vars(parse_arguments()),
    ) as run:
        np.random.seed(run.config['seed'])

        if run.config["split_rules_artifact"] is not None and run.config["coco_path_prefix"] is not None:
            raise ValueError("Only one of split_rules_artifact or coco_path_prefix can be provided.")

        netcdf_path = Path(run.config['netcdf_path'])

        if run.config["split_rules_artifact"] is not None:
            patch_paths = [
                patch_path.relative_to(netcdf_path)
                for patch_path in sorted(list(netcdf_path.glob('**/*.nc')))
            ]
            if run.config["limit_patches"] is not None:
                patch_paths = patch_paths[:run.config["limit_patches"]]
                run.tags = run.tags + ("devtest",)
            split_df, split_rules_name = split_by_split_rules(run.config["split_rules_artifact"], patch_paths)
        elif run.config["coco_path_prefix"] is not None:
            split_df, split_rules_name = split_by_coco_files(run.config["coco_path_prefix"])
        else:
            raise ValueError("Either split_rules_artifact or coco_path_prefix must be provided.")

        # Process each patch
        patch_processors = [
            PixelCounter(),
            PatchVisualizer(),
        ]
        with tqdm(total=len(split_df)) as pbar:
            for target, split in split_df.groupby("target"):
                for i, row in split.iterrows():
                    with netCDF4.Dataset(netcdf_path / row["path"]) as dataset:
                        for patch_processor in patch_processors:
                            patch_processor.process(Path(row["path"]), target, dataset)
                    pbar.update(1)

        if run.config["shuffle"]:
            split_df = split_df.sample(frac=1).reset_index(drop=True)

        # Create a wandb artifact
        create_artifact(run.config, split_rules_name, split_df, patch_processors)


if __name__ == '__main__':
    main()
