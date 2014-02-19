#!/usr/bin/env python
'''
Description     : Module to handle data munging/wrangling
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2014.02.13
Last Modified   : 2014.02.19
Modified By     : Jin Kim jjinking(at)gmail(dot)com
'''

import numpy as np
import pandas as pd
import random
import re
import eda
from sklearn import preprocessing

def standardize_cols(df, cols=None, ignore_binary=True):
    '''
    Standardize selected columns of a df
    if ignore_bingary is True, then do not standardize columns containing binary values
    '''
    # If cols is blank, use all columns in the dataframe
    _cols = cols
    if _cols is None:
        _cols = df.columns

    # Remove binary columns
    if ignore_binary:
        bc = set(eda.find_binary_cols(df))
        _cols = [c for c in _cols if c not in bc]

    df2 = df.copy(deep=True)
    df2[_cols] = preprocessing.scale(df[_cols].values.astype(float))
    return df2

def match_binary_labels(df, ycolname, ylabs=[0, 1], rseed=None):
    '''
    Given a df, match the number of rows with y-labels=0 with the number of rows with y-labels=1
    
    This function should be used when the label counts are so skewed that it's difficult to train a 
    classifier.
    
    In order to downsample from the larger group, only complete rows will be randomly selected.
    If removing incomplete rows in the larger group causes it to be smaller, 
    then the difference will be made up by sampling from the removed incomplete rows.
    '''
    # Make sure ylabs contains only two values
    if len(set(ylabs)) != 2:
        raise ValueError('ylabs should contain two unique values: Found: {0}'.format(ylabs))
    
    # Get y-label series
    ycol = df[ycolname]

    # Check that it's a binary classification problem
    class_counts = ycol.value_counts()
    if len(class_counts) != 2:
        raise ValueError('Number of unique values not equal to 2 in column {0}'.format(ycolname))

    # Check that the y column values match with y labs
    if set(class_counts.index) != set(ylabs):
        raise ValueError('Y labels do not match: Found: \{{0},{1}\} Given: \{{2},{3}\}'
                         .format(class_counts.index[0],
                                 class_counts.index[1],
                                 ylabs[0],
                                 ylabs[1]))
    
    # Divide up the data frame into the two class label groups
    _t0 = df[ycol == ylabs[0]]
    _t1 = df[ycol == ylabs[1]]

    # If the number of rows in both groups are equal, just return the original df
    if _t0.shape[0] == _t1.shape[0]:
        return df
    elif _t0.shape[0] > _t1.shape[0]:
        group_large = _t0
        group_small = _t1
    else:
        group_large = _t1
        group_small = _t0

    random.seed(rseed)

    # TODO: Optimize this part using indices (in a hurry for now)
    # Remove nans
    group_large_dropna = group_large.dropna()
    if group_large_dropna.shape[0] == group_small.shape[0]:
        return pd.concat([group_large_dropna,
                          group_small])
    # Randomly sample from group_large
    elif group_large_dropna.shape[0] > group_small.shape[0]:
        rows = random.sample(group_large_dropna.index, group_small.shape[0])
        return pd.concat([group_large_dropna.ix[rows],
                          group_small])
    # group_large now too small, must add additional missing data
    else:
        dropped_idx = list(set(group_large.index) - set(group_large_dropna.index))
        rows = random.sample(dropped_idx, group_small.shape[0] - group_large_dropna.shape[0])
        return pd.concat([group_large_dropna,
                          group_large.ix[rows],
                          group_small])

def scale_down(strval, mvleft=6):
    '''
    Scale down a string numeric value by moving the decimal to the left
    mvleft times
    '''
    # Raise error for negative mvleft values
    if mvleft < 1:
        raise ValueError('mvleft must be an integer greater than 0\n')

    # Raise error if strval is non-numeric string
    if type(strval) == str and not re.search(r'^[0-9.]+$', strval):
        raise ValueError('Could not scale non-numeric string {0}\n'.format(strval))

    # If value is None or np.nan, just return the value
    if type(strval) != str:
        if pd.isnull(strval):
            return strval
        elif np.isnan(strval):
            return strval
        else:
            raise ValueError('Could not scale value {0} of type {1}\n'.format(strval, type(strval)))

    # strval cannot have more than one decimal
    if strval.count('.') > 1:
        raise ValueError('Value contains more than one decimal {0}\n'.format(strval))

    # Split up the string value by decimal
    strvals = strval.split('.')
    strval_left = strvals[0]
    strval_right = strvals[1] if len(strvals) == 2 else '0'
    strval_left_len = len(strval_left)
    if strval_left_len <= mvleft:
        strval_left = ''.join(['0' for p in range(mvleft - strval_left_len + 1)]) + strval_left
    return float('.'.join([strval_left[:-mvleft],
                           strval_left[-mvleft:] + strval_right]))

def scale_down_cols(df, cols, mvleft=6):
    '''
    Scale down the values in several columns by moving the decimal in each value mvleft times left
    '''
    df2 = df.copy(deep=True)
    for col in cols:
        df2[col] = df2[col].apply(scale_down, mvleft=mvleft)
    return df2
