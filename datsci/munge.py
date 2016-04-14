"""Module to handle data munging/wrangling

"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2014.02.13
# Last Modified   : 2016.04.14
#
# License         : MIT

import csv
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
        writer.writerow(reader.next())
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


def one_hot_encode_features(df, columns=[]):
    """Create one-hot encoded features and remove original

    If columns is empty list, then all features will be one-hot encoded
    """
    # Determine which columns to one-hot encode
    if not columns:
        columns = df.columns
    col = columns[0]
    ohe_df = pd.get_dummies(df[col], prefix='onehot_' + col, prefix_sep='_')
    for col in columns[1:]:
        ohe_df = ohe_df.join(
            pd.get_dummies(df[col], prefix='onehot_' + col, prefix_sep='_'))

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
