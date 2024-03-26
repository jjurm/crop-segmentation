# Based on:
# https://github.com/DanilZherebtsov/verstack/blob/dcb777f5c2558f9d046072dbcf9e41d6b7b8a216/verstack/stratified_continuous_split.py

from collections import Counter

import numpy as np


def find_neighbors_in_two_lists(keys_with_single_value, list_to_find_neighbors_in):
    """Iterate over each item in first list to find a pair in the second list"""
    neighbors = []
    for i in keys_with_single_value:
        for j in [x for x in list_to_find_neighbors_in if x != i]:
            if i + 1 == j:
                neighbors.append(i)
                neighbors.append(j)
            if i - 1 == j:
                neighbors.append(i)
                neighbors.append(j)
    return neighbors


def no_neighbors_found(neighbors):
    """Check if list is empty"""
    return not neighbors


def find_keys_without_neighbor(neighbors):
    """Find integers in list without pair (consecutive increment of + or - 1 value) in the same list"""
    no_pair = []
    for i in neighbors:
        if i + 1 in neighbors:
            continue
        elif i - 1 in neighbors:
            continue
        else:
            no_pair.append(i)
    return no_pair


def not_need_further_execution(y_binned_count):
    """Check if there are bins with single value counts"""
    return 1 not in y_binned_count.values()


def combine_single_valued_bins(y_binned: np.ndarray):
    """
    Correct the assigned bins if some bins include a single value (cannot be split).

    Find bins with single values and:
        - try to combine them to the nearest neighbors within these single bins
        - combine the ones that do not have neighbors among the single values with
        the rest of the bins.

    Args:
        y_binned (array): original y_binned values.

    Returns:
        y_binned (array): processed y_binned values.

    """
    # count number of records in each bin
    y_binned_count = dict(Counter(y_binned))

    if not_need_further_execution(y_binned_count):
        return y_binned

    # combine the single-valued-bins with nearest neighbors
    keys_with_single_value = [k for k, v in y_binned_count.items() if v == 1]

    # first look for neighbors among other single keys
    neighbors1 = find_neighbors_in_two_lists(keys_with_single_value, keys_with_single_value)
    if no_neighbors_found(neighbors1):
        # then look for neighbors among other available keys
        neighbors1 = find_neighbors_in_two_lists(keys_with_single_value, y_binned_count.keys())
    # now process keys for which no neighbor was found
    leftover_keys_to_find_neighbors = list(set(keys_with_single_value).difference(neighbors1))
    neighbors2 = find_neighbors_in_two_lists(leftover_keys_to_find_neighbors, y_binned_count.keys())
    neighbors = sorted(list(set(neighbors1 + neighbors2)))

    # split neighbors into groups for combining
    # only possible when neighbors are found
    if len(neighbors) > 0:
        splits = int(len(neighbors) / 2)
        neighbors = np.array_split(neighbors, splits)
        for group in neighbors:
            val_to_use = group[0]
            for val in group:
                y_binned = np.where(y_binned == val, val_to_use, y_binned)
                keys_with_single_value = [x for x in keys_with_single_value if x != val]

    # --------------------------------------------------------------------------------
    # now combine the leftover keys_with_single_values with the rest of the bins
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    for val in keys_with_single_value:
        nearest = find_nearest([x for x in y_binned if x not in keys_with_single_value], val)
        ix_to_change = np.where(y_binned == val)[0][0]
        y_binned[ix_to_change] = nearest

    return y_binned
