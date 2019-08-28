"""Module to handle data munging/wrangling
"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2014.02.13
# Last Modified   : 2016.04.15
#
# License         : MIT

import csv
import numpy as np
import pandas as pd
import sys
from datetime import datetime

from . import dataio


def remove_null_big_data(fname_in, fname_out, delim=','):
    """Remove all rows containing any null values
    """
    with dataio.fopen(fname_in, 'r') as fi, dataio.fopen(fname_out, 'w') as fo:
        reader = csv.reader(fi, delimiter=delim)
        writer = csv.writer(fo, delimiter=delim)
        writer.writerow(next(reader))
        for row in reader:
            if '' in row:
                continue
            writer.writerow(row)


def remove_col_big_data(
        fname_in, fname_out, indices, delimiter=',', verbose=None):
    """Remove a column given the given column index idx (0-based)
    if progress_int can be set to an integer to set an interval to output
    number of rows seen every given interval
    """
    indices = set(indices)
    with dataio.fopen(fname_in, 'r') as fi, dataio.fopen(fname_out, 'w') as fo:
        reader = csv.reader(fi, delimiter=delimiter)
        writer = csv.writer(fo, delimiter=delimiter)
        for t, row in enumerate(reader):
            # Output progress
            if verbose is not None and t % verbose == 0:
                sys.stdout.write(
                    '{}\tencountered: {}\n'.format(datetime.now(), t))
            writer.writerow([x for i, x in enumerate(row) if i not in indices])

def one_hot_encode_series_of_lists(feature_col, list_values=None):
    """
    Take in a pd.Series `feature_col` containing `list_values`, and return a
    pd.DataFrame containing one-hot encoded columns.
    
    If `list_values` is None, then a column for every unique value in
    the series will be created.
    
    Otherwise, a column will be created for each of the values in `list_values`.
    Also, an additional default column will be created to hold values that
    did not belong to any of the `list_values`. Note, if a row in `feature_col`
    contains multiple values that are not in `list_values`, there will only be a
    single 1 in the default column.
    
    """
    if list_values is None:
        seen_vals = dict()
    else:
        # Initialize default column
        default_colname = feature_col.name + '_default'
        seen_vals = {default_colname: []}
        for val in list_values:
            seen_vals[feature_col.name + '_' + val] = []
    idx = 0
    prefix = feature_col.name
    for vals in feature_col:
        for val in vals:
            val_name = prefix + '_' + val
            if val_name not in seen_vals and list_values is None:
                seen_vals[val_name] = ([0] * idx) + [1]
            else:
                if val_name not in seen_vals:
                    val_name = default_colname
                for i in range(idx - len(seen_vals[val_name])):
                    seen_vals[val_name].append(0)
                if len(seen_vals[val_name]) - 1 < idx:
                    seen_vals[val_name].append(1)
        idx += 1
    # Fill out remaining zeros
    for k,v in seen_vals.items():
        for _ in range(idx - len(v)):
            v.append(0)
    return pd.DataFrame(seen_vals)

def one_hot_encode_list_cols(df, columns=[]):
    """Create one-hot encoded features and remove original
    *Note* if columns is empty list, then all features will be one-hot encoded
    """
    # Determine which columns to one-hot encode
    if not columns:
        columns = df.columns
    
    ohe_df = one_hot_encode_series_of_lists(df[columns[0]])
    for col in columns[1:]:
        ohe_df = ohe_df.join(one_hot_encode_series_of_lists(df[col]))

    # Attach to non-one-hot encoded columns
    non_ohe_cols = []
    columns_set = set(columns)
    for c in df.columns:
        if c not in columns_set:
            non_ohe_cols.append(c)

     # If all columns are one-hot encoded, return them
    if not non_ohe_cols:
        return ohe_df

    # Otherwise, join the two dfs
    return df[non_ohe_cols].join(ohe_df)

def one_hot_encode(df, columns=[]):
    """Create one-hot encoded features and remove original
    *Note* if columns is empty list, then all features will be one-hot encoded
    """
    # Determine which columns to one-hot encode
    if not columns:
        columns = df.columns
    col = columns[0]
    ohe_df = pd.get_dummies(df[col], prefix=col, prefix_sep='_')
    for col in columns[1:]:
        ohe_df = ohe_df.join(
            pd.get_dummies(df[col], prefix=col, prefix_sep='_'))

    # Attach to non-one-hot encoded columns
    non_ohe_cols = []
    columns_set = set(columns)
    for c in df.columns:
        if c not in columns_set:
            non_ohe_cols.append(c)

    # If all columns are one-hot encoded, return them
    if not non_ohe_cols:
        return ohe_df

    # Otherwise, join the two dfs
    return df[non_ohe_cols].join(ohe_df)


def remove_duplicates(df):
    """Remove duplicate rows

    *Note* X.T.remove_duplicates() keeps incurring stack overflow error
    when X has many features, typically in the 100's
    """
    cache = set()

    def is_unique(row):
        rowstr = row.to_string()
        if rowstr in cache:
            return False
        cache.add(rowstr)
        return True

    return df[df.apply(is_unique, axis=1)]


def balanced_subsets(
        X_train, y_train, subsample=1.0, shuffle=True, labels=(0, 1)):
    """Given unbalanced binary target values, create balanced subsets
    that can be used to train multiple models, which can then be used
    for ensembling.

    The last subset contains remainders, so it is bigger than the
    other subsets. Therefore, the last subset may not be balanced.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Input samples

    y_train : pandas.Series
        Target values

    subsample : float, (0.0, 1.0]
        Sampling rate of the minority class within the balanced subsets

    shuffle : bool
        If True, Shuffle the data before creating subsets

    labels : list or tuple of size 2
        Contains unique values class labels

    Returns
    -------
    subsamples : list of (pd.DataFrame, pd.Series) 2-tuples
        Subsamples of X and y containing 'balanced' labels.
        ex: [(X1, y1), (X2, y2), ...]
    """
    # Save column labels
    features = X_train.columns
    target_name = y_train.name

    # Get numpy.ndarray values
    X = X_train.values
    y = y_train.values

    # Find indices of y labels
    idx = {
        False: np.argwhere(y == labels[0]).reshape(-1),
        True:  np.argwhere(y == labels[1]).reshape(-1)
    }

    # Count number of samples for each class label
    n_label = {
        False: len(idx[False]),
        True:  len(idx[True])
    }

    # Counts of majority and minority y labels
    label_maj = True if n_label[False] < n_label[True] else False
    label_min = not label_maj

    # Shuffle the samples (just the majority labels)
    if shuffle:
        idx[label_maj] = np.random.permutation(idx[label_maj])

    # Create subsamples
    subsamples = []
    batch_size = int(subsample * n_label[label_min])
    batch_size_combined = batch_size * 2  # combined batch size
    L = int(n_label[label_maj] / batch_size)
    a = 0
    b = batch_size
    for i in range(L - 1):
        # Minority labels
        batch_min_idx = np.random.choice(
            idx[label_min], size=batch_size, replace=False)
        batch_min_X = X[batch_min_idx]
        batch_min_y = y[batch_min_idx]

        # Majority labels
        batch_maj_idx = idx[label_maj][a:b]
        a, b = b, b + batch_size
        batch_maj_X = X[batch_maj_idx]
        batch_maj_y = y[batch_maj_idx]

        # Intersperse the labels
        batch_X = np.vstack((batch_min_X, batch_maj_X))
        batch_y = np.hstack((batch_min_y, batch_maj_y))
        mixed_idx = np.random.permutation(batch_size_combined)
        subsamples.append((
            pd.DataFrame(batch_X[mixed_idx], columns=features),
            pd.Series(batch_y[mixed_idx], name=target_name)
        ))

    # -- Last subset containing remainders

    # Majority labels
    batch_maj_idx = idx[label_maj][a:]
    batch_maj_X = X[batch_maj_idx]
    batch_maj_y = y[batch_maj_idx]

    # Minority labels
    last_maj_batch_size = len(batch_maj_idx)
    if last_maj_batch_size == batch_size:
        last_min_batch_size = batch_size
    elif last_maj_batch_size < n_label[label_min]:
        last_min_batch_size = last_maj_batch_size
    else:
        last_min_batch_size = n_label[label_min]
    batch_min_idx = np.random.choice(
        idx[label_min], size=last_min_batch_size, replace=False)
    batch_min_X = X[batch_min_idx]
    batch_min_y = y[batch_min_idx]

    # Intersperse the labels
    batch_X = np.vstack((batch_min_X, batch_maj_X))
    batch_y = np.hstack((batch_min_y, batch_maj_y))
    mixed_idx = np.random.permutation(len(batch_y))
    subsamples.append((
        pd.DataFrame(batch_X[mixed_idx], columns=features),
        pd.Series(batch_y[mixed_idx], name=target_name)
    ))
    return subsamples
