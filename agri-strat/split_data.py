#!/usr/bin/env python
"""This file reads a yaml file from a wandb artifact describing the splits over patches and splits the data into
train, val and test sets, writing the results of each split to a separate txt file.

Information about each patch (year, tile, patch_x, patch_y) is taken from the file name, e.g. 2019_31TCJ_patch_20_29.nc
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import swifter

import wandb
from utils.splits.coco import split_by_coco_files
from utils.splits.patch.elevation import PatchElevationStats
from utils.splits.patch.filename_attrs import PatchFilenameAttrs
from utils.splits.patch.geometry import PatchGeometry
from utils.splits.patch.patch_processor import PatchProcessor, PatchApplyFn
from utils.splits.patch.pixel_counter import ClassPixelCounts, ClassPixelCountsPerSplit
from utils.splits.split_rules import split_by_split_rules, random_splits


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_rules_artifact', type=str, required=False,
                        help='Wandb artifact of the \'split_rules\' type.')
    parser.add_argument('--coco_path', type=str, default=None, required=False,
                        help='Path to the COCO files. Default $COCO_PATH or "dataset/coco_files".')
    parser.add_argument('--coco_prefix', type=str, default=None, required=False)
    parser.add_argument('--netcdf_path', type=str, default=None, required=False,
                        help='Path to the netCDF files. Default $NETCDF_PATH or "dataset/netcdf".')

    parser.add_argument('--seed', type=int, default=0, required=False,
                        help='Seed for random splits and shuffling. Default: 0.')
    parser.add_argument('--shuffle', action='store_true', default=False, required=False,
                        help='Shuffle the order of patches written out. Default: False.')

    parser.add_argument('--elevation', action='store_true', default=False, required=False,
                        help='Use the elevation data for stratification. Default: False.')
    parser.add_argument("--dem_path", type=str, default=None, required=False,
                        help="Directory to save the SRTM30m data. Default: $DEM_PATH or 'dataset/dem/srtm30'.")

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
        patch_processor.log_to_artifact(split_df, artifact)

    wandb.run.log_artifact(artifact)


def shuffle(split_df):
    return split_df.sample(frac=1).reset_index(drop=True)


def main():
    swifter.set_defaults(
        force_parallel=True,
        allow_dask_on_strings=True,
    )

    with wandb.init(
            project='agri-strat',
            job_type='generate-split',
            config=vars(parse_arguments()),
    ) as run:
        np.random.seed(run.config['seed'])

        if run.config["split_rules_artifact"] is not None and run.config["coco_prefix"] is not None:
            raise ValueError("Only one of split_rules_artifact or coco_path_prefix can be provided.")

        netcdf_path = Path(run.config['netcdf_path'] or os.getenv("NETCDF_PATH", "dataset/netcdf"))

        # List all patches and split rules
        if run.config["split_rules_artifact"] is not None:
            patch_paths = [
                patch_path.relative_to(netcdf_path)
                for patch_path in sorted(list(netcdf_path.glob('**/*.nc')))
            ]
            if len(patch_paths) == 0:
                raise ValueError(f"No patches found in {netcdf_path}.")
            if run.config["limit_patches"] is not None:
                patch_paths = patch_paths[:run.config["limit_patches"]]
                run.tags = run.tags + ("devtest",)
            split_df, rules_to_split, split_rule_defs, split_rules_name = split_by_split_rules(
                run.config["split_rules_artifact"], patch_paths)
        elif run.config["coco_prefix"] is not None:
            coco_path = Path(run.config["coco_path"] or os.getenv("COCO_PATH", "dataset/coco_files"))
            split_df, split_rules_name = split_by_coco_files(coco_path / run.config["coco_prefix"])
            rules_to_split = {}
            split_rule_defs = None
        else:
            raise ValueError("Either split_rules_artifact or coco_path_prefix must be provided.")
        split_df.set_index("path", inplace=True)

        # Process each patch
        print("Pre-processing patches...")
        patch_preprocessors = [
            PatchFilenameAttrs(),
            PatchGeometry(),
            (pixel_counter := ClassPixelCounts()),
        ]
        if run.config["elevation"]:
            patch_preprocessors.append(PatchElevationStats(
                srtm_dataset_path=Path(run.config["dem_path"] or os.getenv("DEM_PATH", "dataset/dem/srtm30"))
            ))
        preprocess_fn = PatchApplyFn(patch_preprocessors, with_netcdf_file=True, netcdf_path=netcdf_path)
        split_df = split_df.swifter.apply(preprocess_fn, axis=1)

        # Now perform random splits
        random_splits(split_df, rules_to_split, split_rule_defs)

        # Process each patch again
        print("Post-processing patches...")
        patch_postprocessors = [
            ClassPixelCountsPerSplit(classes=pixel_counter.classes),
        ]
        postprocess_fn = PatchApplyFn(patch_postprocessors)
        split_df = split_df.swifter.apply(postprocess_fn, axis=1)

        if run.config["shuffle"]:
            split_df = shuffle(split_df)

        # Create a wandb artifact
        create_artifact(run.config, split_rules_name, split_df, patch_preprocessors + patch_postprocessors)


if __name__ == '__main__':
    main()
