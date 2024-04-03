# Partly based on:
# https://github.com/DanilZherebtsov/verstack/blob/dcb777f5c2558f9d046072dbcf9e41d6b7b8a216/verstack/stratified_continuous_split.py

from collections import Counter
from functools import partial
from typing import cast

import numpy as np
import pandas as pd
# noinspection PyProtectedMember
from pandas.api.types import is_numeric_dtype


def _floor_median_from_counts(values: np.ndarray[np.integer], counts: np.ndarray | list) -> np.integer:
    """
    Find floor(median) of an array of counts.
    Adjusted from https://gist.github.com/alexandru-dinu/4c6133202d8d379066994f5d11446e9d
    :param values: array of values, must be sorted
    :param counts: where counts[i] is number of occurrences of values[i]
    """
    cf = np.cumsum(counts)
    n = cf[-1]

    # get the left and right buckets
    # of where the midpoint falls,
    # accounting for both even and odd lengths
    left = (n // 2 - 1) < cf
    right = (n // 2) < cf

    # median is the midpoint value (which falls in the same bucket)
    if n % 2 == 1 or (left == right).all():
        values_: np.ndarray[np.integer] = values[right]
        return values_[0]
    # median is the mean between the mid adjacent buckets
    else:
        return np.mean(values[left | right][:2], dtype=np.integer)


def _combine_single_valued_bins(y_binned, bins: pd.Index) -> list:
    """
    Correct the assigned bins if some bins include a single value (cannot be split).
    Tries to combine singleton bins to the nearest neighbors within these single bins, giving preference to other
    singleton bins.
    :param y_binned: original y_binned values
    :param bins: list of all values that elements o y_binned can take
    :return: processed y_binned values
    """

    y_bin_indices = np.array([bins.get_loc(x) for x in y_binned])
    n_bins = len(bins)

    for distance in range(1, n_bins):
        # count number of records in each bin
        counts = dict(Counter(y_bin_indices))
        singleton_bins = [bin_i for bin_i, count in counts.items() if count == 1]
        if len(singleton_bins) == 0:
            break

        # find a neighbour for each singleton bin (at `distance` bins away)
        neighbors = set()
        for bin_i in singleton_bins:
            if bin_i in neighbors:
                continue
            options = [
                (counts[bin_j] > 1, bin_j)  # prefer other singleton bins
                for bin_j in [bin_i - distance, bin_i + distance]
                if bin_j in counts
            ]
            if options:
                _, bin_j = min(options)
                neighbors.add(bin_i)
                neighbors.add(bin_j)

        # combine the bins
        if len(neighbors) > 0:
            n_splits = int(len(neighbors) / 2)
            for group in np.array_split(sorted(neighbors), n_splits):
                target_bin_index = _floor_median_from_counts(
                    cast(group, np.ndarray[np.integer]),
                    [counts[bin_i] for bin_i in group])
                for val in group:
                    y_bin_indices = np.where(y_bin_indices == val, target_bin_index, y_bin_indices)

    return [bins[x] for x in y_bin_indices]


def _combine_bins(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Given a dataframe and a column, returns the same dataframe with singleton values in the column combined.
    :param df: a DataFrame
    :param column: column name
    :return: DataFrame with the column binned
    """
    features_combined = _combine_single_valued_bins(df[column], df[column].dtype.categories)
    return df.assign(**{column: features_combined})


def _combine_single_valued_bins_multilevel(df: pd.DataFrame, stratify_on: list) -> pd.DataFrame:
    """
    Given a dataframe, converts all columns to categorical, and combines single-valued bins for each column.

    The order of stratify_on is relevant. First features are treated more important, meaning that with sparse bins,
    latter features will be combined first.
    Combining single-valued bins of a feature X is done in a group of rows that have the same value of all higher
    features, and behaves as if lower features were not present.
    :param df: a DataFrame
    :param stratify_on: list of features to stratify on, a subset of df.columns
    :return: DataFrame with columns corresponding to stratify_on features
    """

    grouped = df.groupby(by=lambda x: 0)
    for feature_column in stratify_on:
        # For each group, bin the feature_column, then group by the binned column
        fn_per_group = partial(_combine_bins, column=feature_column)
        grouped = grouped.apply(fn_per_group, include_groups=False).groupby(by=feature_column)

    # Since the last grouping has no more features to bin, unwrap the index back to the dataframe
    df_binned_combined = grouped.apply(lambda x: x, include_groups=False) \
        .reset_index(level=stratify_on) \
        .reset_index(level=0, drop=True) \
        .reindex_like(df)
    return df_binned_combined


def _bin_features(feature: pd.Series, n_bins: int = None) -> pd.Series:
    """
    Given a feature, returns the binned feature if continuous/numerical. The returned values potentially form
    single-valued bins.
    :param feature: a Series
    :param n_bins: number of bins
    :return: the binned feature
    """
    if is_numeric_dtype(feature.dtype):
        n_bins = n_bins or min(100, len(feature) // 20)
        features_binned = pd.cut(feature, bins=n_bins, labels=np.arange(n_bins))
        return pd.Series(features_binned, index=feature.index)
    else:
        raise ValueError(f"Feature {feature.name} to be stratified is of unsupported type {feature.dtype}.")


def get_features_for_stratifying(df: pd.DataFrame, stratify_on: list) -> pd.DataFrame:
    """
    Given a dataframe, returns a dataframe with all columns converted to categorical and binned.
    :param df: a DataFrame
    :param stratify_on: list of features to stratify on, a subset of df.columns, each feature in the form
        `{column: "feature_name", n_bins: 10}`
    :return: DataFrame with stratified features
    """
    y_df = pd.DataFrame(index=df.index)
    for feature in stratify_on:
        feature_column = feature["column"]
        y_df[feature_column] = _bin_features(df[feature_column], feature.get("n_bins"))
    return y_df


def combine_features_for_stratifying(y_df: pd.DataFrame) -> pd.Series:
    """
    Combine single-valued bins across multiple levels of features, then concatenate the features.
    :param y_df: a DataFrame
    :return: Series with concatenated stratified features
    """
    y_df_combined = _combine_single_valued_bins_multilevel(y_df, stratify_on=y_df.columns)
    return y_df_combined.apply(lambda x: '__'.join(map(str, x)), axis=1)
