#!/usr/bin/env python
"""This file reads a yaml file from a wandb artifact describing the splits over patches and splits the data into
train, val and test sets, writing the results of each split to a separate txt file.

Information about each patch (year, tile, patch_x, patch_y) is taken from the file name, e.g. 2019_31TCJ_patch_20_29.nc
"""

import argparse
from pathlib import Path

import numpy as np

import wandb
from utils.splits.coco import split_by_coco_files
from utils.splits.split_rules import split_by_split_rules
from utils.splits.visualize import create_split_visualization


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
    return parser.parse_args()


def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    with wandb.init(
            project='agri-strat',
            job_type='generate-split',
            config=vars(args),
    ) as run:
        if args.split_rules_artifact is not None and args.coco_path_prefix is not None:
            raise ValueError("Only one of split_rules_artifact or coco_path_prefix can be provided.")

        data_path = Path(args.netcdf_path)
        patch_paths = [
            patch_path.relative_to(data_path)
            for patch_path in sorted(list(data_path.glob('**/*.nc')))
        ]

        if args.split_rules_artifact is not None:
            split_df, split_rules_name = split_by_split_rules(args.split_rules_artifact, patch_paths)
        elif args.coco_path_prefix is not None:
            split_df, split_rules_name = split_by_coco_files(args.coco_path_prefix)
        else:
            raise ValueError("Either split_rules_artifact or coco_path_prefix must be provided.")

        gpkg_path = create_split_visualization(split_df, data_path)

        if args.shuffle:
            split_df = split_df.sample(frac=1).reset_index(drop=True)

        # Log number of patches per split
        patch_counts = split_df['target'].value_counts()
        print(patch_counts.to_string(header=False))

        # Write a txt for each split
        splits_dir = Path(run.dir) / 'splits'
        splits_dir.mkdir(parents=True, exist_ok=True)
        for target, target_df in split_df.groupby('target'):
            target_file_path = splits_dir / f"{target}.txt"
            target_df.drop(columns='target').to_csv(target_file_path, index=False, header=False)
            print(f"Split {target} written to {target_file_path}.")

        artifact_name_prefix = args.artifact_name_prefix or split_rules_name
        artifact = wandb.Artifact(
            name=f"{artifact_name_prefix}_split",
            type="split",
            metadata={
                "split_rules_name": split_rules_name,
            },
        )
        artifact.add_dir(splits_dir.as_posix(), name="splits")
        artifact.add(wandb.Table(dataframe=patch_counts.to_frame().reset_index()), name="patch_counts")
        artifact.add(gpkg_path.as_posix(), name="splits_polygons")
        run.log_artifact(artifact)


if __name__ == '__main__':
    main()
