"""This file reads a yaml file from a wandb artifact describing the splits over patches and splits the data into
train, val and test sets, writing the results of each split to a separate txt file.

Information about each patch (year, tile, patch_x, patch_y) is taken from the file name, e.g. 2019_31TCJ_patch_20_29.nc
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_rules_artifact', type=str, required=True,
                        help='Wandb artifact of the \'split_rules\' type.')
    parser.add_argument('--netcdf_path', type=str, default='dataset/netcdf', required=False,
                        help='Path to the netCDF files. Default "dataset/netcdf".')
    parser.add_argument('--seed', type=int, default=0, required=False,
                        help='Seed for random splits and shuffling. Default: 0.')
    parser.add_argument('--shuffle', action='store_true', default=False, required=False,
                        help='Shuffle the order of patches written out. Default: False.')
    return parser.parse_args()


class SetMeOneTimeTarget:
    def __init__(self, rule_loc):
        self.value = None
        self.was_set = False
        self.rule_loc = rule_loc

    def set(self, value):
        if self.was_set:
            raise ValueError(f"The split rule {self.rule_loc} sets the target more than once. Rules should set the "
                             f"target only once.")
        self.was_set = True
        self.value = value

    def get(self):
        if not self.was_set:
            raise ValueError(f"The split rule {self.rule_loc} did not set the target. Each rule should set one of "
                             f"'target', 'rules' or 'assert_none'.")
        return self.value


def passes_filter(metadata, key, filter_value):
    if isinstance(filter_value, str) or isinstance(filter_value, int):
        filter_value = {"in": [filter_value]}
    if isinstance(filter_value, list):
        filter_value = {"in": filter_value}

    for op, value in filter_value.items():
        if op == "in":
            if metadata[key] not in value:
                return False
        elif op == "min":
            if metadata[key] < value:
                return False
        elif op == "max":
            if metadata[key] > value:
                return False
        else:
            raise ValueError(f"Unknown operator {op} in filter {key}: {filter_value}")
    return True


def recursive_search(split_rules, metadata, rule_loc: tuple[int, ...]) -> str | None:
    """
    Recursively search for the target of a patch based on the split rules.
    :param split_rules: rules from the yaml file
    :param metadata: metadata of the patch
    :param rule_loc: location of the current rule in the yaml file, given as a tuple of indices from outermost
    :return: If the target is found and definite, return the target.
    If the target is found but a random split is to be done later, return rule_loc (list).
    Otherwise, if no target is found, return None.
    """
    for i, rule in enumerate(split_rules):
        inner_rule_loc = rule_loc + (i,)
        target = SetMeOneTimeTarget(inner_rule_loc)
        for key, filter_value in rule.items():
            if key in metadata.keys():
                if not passes_filter(metadata, key, filter_value):
                    break
            elif key == "target":
                if isinstance(filter_value, str):
                    target.set(filter_value[0])
                else:
                    target.set(inner_rule_loc)
            elif key == "rules":
                target.set(recursive_search(filter_value, metadata, inner_rule_loc))
            elif key == "assert_none" and filter_value:
                raise ValueError(f"Patch {metadata['path']} reached an assert_none rule {inner_rule_loc}")
            else:
                raise ValueError(f"Unknown key {key} in split rule {inner_rule_loc}")
        else:
            return target.get()
    return None


def find_target_of_patch(patch_path, split_rules):
    year, tile, _, patch_x, patch_y = patch_path.stem.split('_')
    metadata = {
        'year': int(year),
        'tile': tile,
        'patch_x': int(patch_x),
        'patch_y': int(patch_y),
        'path': patch_path,
    }
    return recursive_search(split_rules, metadata, ())


def get_rule_by_loc(split_rules, rule_loc: tuple[int, ...]):
    rule = split_rules
    for i in rule_loc:
        rule = rule["rules"][i]
    return rule


def calculate_split_boundaries(targets):
    # If any of the targets have unspecified 'split' property, the remaining fraction is split evenly among them
    split_fractions = [target["split"] for target in targets if "split" in target]
    fraction_unspecified = 1 - sum(split_fractions)
    n_unspecified = len(targets) - len(split_fractions)
    fraction_boundaries = np.cumsum([
        target["split"] if "split" in target else fraction_unspecified / n_unspecified
        for target in targets
    ])
    return fraction_boundaries


def random_split(targets, patch_paths, rule_loc):
    fraction_boundaries = calculate_split_boundaries(targets)
    assert fraction_boundaries[-1] < 1 or np.isclose(fraction_boundaries[-1], 1), (
        f"Split fractions should sum to 1 but sume to {fraction_boundaries[-1]} in rule {rule_loc}")
    boundaries = np.rint(fraction_boundaries * len(patch_paths)).astype(int)
    num_patches_per_split = [
        (target["target"], num_patches) for target, num_patches in zip(targets, np.diff([0, *boundaries]))
    ]
    print(f"Rule {rule_loc}: Splitting {len(patch_paths)} patches into {num_patches_per_split}")
    df = pd.DataFrame({
        "path": patch_paths,
        "target": None,
    })
    indices = np.random.permutation(df.index)
    for target, idx_low, idx_high in zip(targets, [0, *boundaries], boundaries):
        df.loc[indices[idx_low:idx_high], "target"] = target["target"]
    # Only return entries with a target
    return df.dropna(subset=["target"])


def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    with wandb.init(
            project='agri-strat',
            job_type='generate-split',
            config=vars(args),
    ) as run:

        split_rules_artifact = run.use_artifact(args.split_rules_artifact, type='split_rules')
        split_rules_filename = split_rules_artifact.file()
        with open(split_rules_filename, 'r') as file:
            split_rules = yaml.safe_load(file)

        # Accumulate patches and targets (to later build a dataframe)
        split_rows = []

        # The following is a dict of rules where a random split needs to be done after accumulating the patches.
        rules_to_split = {}  # {rule_loc: [patch_paths]}

        data_path = Path(args.netcdf_path)
        patch_paths = sorted(list(data_path.glob('**/*.nc')))

        for patch_path in patch_paths:
            relative_path = patch_path.relative_to(data_path)
            target = find_target_of_patch(relative_path, split_rules["rules"])
            if target is not None:
                if isinstance(target, str):
                    split_rows.append((relative_path, target))
                else:
                    assert isinstance(target, tuple)
                    if target not in rules_to_split:
                        rules_to_split[target] = []
                    rules_to_split[target].append(relative_path)
        split_df = pd.DataFrame.from_records(split_rows, columns=['path', 'target'])

        # Now perform random splits
        for rule_loc, patch_paths in rules_to_split.items():
            targets = get_rule_by_loc(split_rules, rule_loc)["target"]
            assert isinstance(targets, list)  # If string, it would have been assigned a target by find_target_of_patch

            df = random_split(targets, patch_paths, rule_loc)
            split_df = pd.concat([split_df, df])

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

        artifact = wandb.Artifact(
            name=f"{split_rules_artifact.collection.name}_split",
            type="split",
            metadata={
                "split_rules_name": split_rules_artifact.collection.name,
            },
        )
        artifact.add_dir(splits_dir.as_posix(), name="splits")
        artifact.add(wandb.Table(dataframe=patch_counts.to_frame().reset_index()), name="patch_counts")
        run.log_artifact(artifact)


if __name__ == '__main__':
    main()
