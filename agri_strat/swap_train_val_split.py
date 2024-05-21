#!/usr/bin/env python
"""
Given a split artifact with train, val, test splits, this script will produce an artifact with the train and val splits
swapped. This is useful for training a model on the val set, such as the irreducible-loss model from the RHO-LOSS paper.
"""

import argparse
import shutil
from pathlib import Path

import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_artifact', type=str, default=None, required=False,
                        help='Wandb artifact of type \'split\'.')
    return parser.parse_args()


def map_splits(split: str) -> str:
    if split == 'train':
        return 'val'
    elif split == 'val':
        return 'train'
    else:
        return split


def main():
    with wandb.init(
            project='agri-strat',
            job_type='swap-split',
            config=vars(parse_arguments()),
    ) as run:
        if run.config.split_artifact is None:
            raise ValueError("split_artifact is required.")

        input_artifact = wandb.run.use_artifact(run.config.split_artifact, type='split')
        input_dir = Path(input_artifact.download())

        artifact = wandb.Artifact(
            name=f"{input_artifact.collection.name}_swap",
            type="split",
            metadata=input_artifact.metadata,
        )
        output_dir = Path(wandb.run.dir)

        populate_artifact(input_artifact, artifact, input_dir, output_dir)

        wandb.run.log_artifact(artifact)


def populate_artifact(input_artifact, output_artifact, input_dir, output_dir):
    (output_dir / "splits").mkdir(parents=True, exist_ok=True)

    # noinspection PyUnresolvedReferences
    patch_counts = input_artifact.get("patch_counts").get_dataframe()
    for i, row in patch_counts.iterrows():
        split = row["target"]

        # splits/*.txt
        shutil.copy(
            input_dir / "splits" / f"{split}.txt",
            output_dir / "splits" / f"{map_splits(split)}.txt",
        )

        # pixel_counts_* table
        output_artifact.add(input_artifact.get(f"pixel_counts_{split}"), name=f"pixel_counts_{map_splits(split)}")

        # TODO consider adding split_polygons.gpkg

    output_artifact.add_dir((output_dir / "splits").as_posix(), name="splits")

    patch_counts["target"] = patch_counts["target"].apply(map_splits)
    output_artifact.add(wandb.Table(dataframe=patch_counts), name="patch_counts")


if __name__ == '__main__':
    main()
