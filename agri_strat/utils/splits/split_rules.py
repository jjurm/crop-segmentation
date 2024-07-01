import numpy as np
import pandas as pd
import wandb
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

from utils.splits.stratified_continuous_split import get_features_for_stratifying, combine_features_for_stratifying


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
                    target.set(filter_value)
                else:
                    target.set(inner_rule_loc)
            elif key == "rules":
                target.set(recursive_search(filter_value, metadata, inner_rule_loc))
            elif key == "assert_none" and filter_value:
                raise ValueError(f"Patch {metadata['path']} reached an assert_none rule {inner_rule_loc}")
            elif key == "stratify_on":
                pass
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


def random_split(targets: list, patch_paths, split_df, rule_loc):
    fraction_boundaries = calculate_split_boundaries(targets)
    assert fraction_boundaries[-1] < 1 or np.isclose(fraction_boundaries[-1], 1), (
        f"Split fractions should sum to 1 but sume to {fraction_boundaries[-1]} in rule {rule_loc}")
    boundaries = np.rint(fraction_boundaries * len(patch_paths)).astype(int)
    num_patches_per_split = [
        (target["target"], num_patches) for target, num_patches in zip(targets, np.diff([0, *boundaries]))
    ]
    print(f"Rule {rule_loc}: Splitting {len(patch_paths)} patches into {num_patches_per_split}")

    indices = np.random.permutation(patch_paths)
    for target, idx_low, idx_high in zip(targets, [0, *boundaries], boundaries):
        split_df.loc[indices[idx_low:idx_high], "target"] = target["target"]


def stratify_dataset(df: pd.DataFrame, indices: pd.Index, stratify_on: list, splits: list[tuple[str, float]],
                     seed=None, binned_features_column_prefix=None):
    """
    Stratify a dataset based on the given columns, adding a column "target" to the DataFrame inplace.
    :param df: DataFrame to stratify
    :param indices: indices of the rows to stratify, all other rows are ignored
    :param stratify_on: list of features to stratify on, a subset of df.columns, each feature in the form
        `{column: "feature_name", n_bins: 10}`
    :param splits: list of tuples (target, split_fraction) where split_fraction is the fraction of the dataset
        that should be assigned to the target
    :param seed: random seed
    :param binned_features_column_prefix: when set, adds the binned features to df with the given prefix
    """
    generator = np.random.default_rng(seed)
    remaining_indices = indices
    remaining_ratio = 1.0

    # convert numerical features to binned features
    y_df = get_features_for_stratifying(df.loc[indices], stratify_on)

    for split, split_ratio in splits:
        train_size = split_ratio / remaining_ratio
        if train_size < 1:
            strat_split = StratifiedShuffleSplit(
                n_splits=1,
                train_size=split_ratio / remaining_ratio,
                random_state=generator.integers(0, 2 ** 32))
            y = combine_features_for_stratifying(y_df.loc[remaining_indices])
            for train_index_i, test_index_i in strat_split.split(remaining_indices, y):
                train_index, test_index = remaining_indices[train_index_i], remaining_indices[test_index_i]
                break
            else:
                assert False, "StratifiedShuffleSplit did not return any splits."
        else:
            train_index, test_index = remaining_indices, pd.Index([])
        df.loc[train_index, "target"] = split
        remaining_indices = test_index
        remaining_ratio -= split_ratio

    # add binned features to the DataFrame
    if binned_features_column_prefix:
        for column in y_df.columns:
            new_column = f"{binned_features_column_prefix}_{column}"
            df.loc[indices, new_column] = y_df.loc[indices, column]


def random_splits(split_df, rules_to_split, split_rule_defs, seed=None):
    generator = np.random.default_rng(seed)
    print("Performing random splits...")
    for rule_loc, patch_paths in sorted(rules_to_split.items()):
        rule = get_rule_by_loc(split_rule_defs, rule_loc)
        if "stratify_on" in rule:
            indices = pd.Index(patch_paths)
            splits = [(target["target"], target["split"]) for target in rule["target"]]
            stratify_dataset(
                split_df, indices, rule["stratify_on"], splits,
                seed=generator.integers(0, 2 ** 32),
                binned_features_column_prefix=f"strat_{'_'.join(map(str, rule_loc))}"
            )
        else:
            targets = rule["target"]
            random_split(targets, patch_paths, split_df, rule_loc)
    # Only return entries with a target
    all_patch_count = len(split_df)
    split_df.dropna(subset=["target"], inplace=True)
    if len(split_df) < all_patch_count:
        print(f"Discarded {all_patch_count - len(split_df)} patches without a target (keeping {len(split_df)}).")


def split_by_split_rules(split_rules_artifact_name, patch_paths):
    """
    Generate a DataFrame with columns "path" and "target" based on the split rules.
    :param split_rules_artifact_name:
    :param patch_paths:
    :return: split_df, split_rules_name
    """
    split_rules_artifact = wandb.run.use_artifact(split_rules_artifact_name, type='split_rules')
    split_rules_filename = split_rules_artifact.file()
    with open(split_rules_filename, 'r') as file:
        split_rule_defs = yaml.safe_load(file)
    # Accumulate patches and targets (to later build a dataframe)
    split_rows = []
    # The following is a dict of rules where a random split needs to be done after accumulating the patches.
    rules_to_split = {}  # {rule_loc: [patch_paths]}
    for relative_path in patch_paths:
        target = find_target_of_patch(relative_path, split_rule_defs["rules"])
        if target is not None:
            if isinstance(target, str):
                split_rows.append((relative_path, target))
            else:
                assert isinstance(target, tuple)
                if target not in rules_to_split:
                    rules_to_split[target] = []
                rules_to_split[target].append(relative_path)
                split_rows.append((relative_path, None))
    split_df = pd.DataFrame.from_records(split_rows, columns=['path', 'target'])

    split_rules_name = split_rules_artifact.collection.name
    return split_df, rules_to_split, split_rule_defs, split_rules_name
