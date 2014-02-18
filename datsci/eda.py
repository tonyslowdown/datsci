#!/usr/bin/env python
'''
Description     : Module to handle EDA (Exploratory Data Analysis)
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2014.02.13
Last Modified   : 2014.02.14
Modified By     : Jin Kim jjinking(at)gmail(dot)com
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpltools import style; style.use('ggplot')

def find_uninfo_cols(df):
    '''
    Find uninformative columns
    i.e. columns with all the same values (excluding nulls)
    '''
    counts = df.apply(lambda col: col[~col.isnull()].nunique())
    return list(counts[counts == 1].index)

def find_null_cols(df, frac=.8):
    '''
    Find columns containing >= frac null values
    '''
    null_fracs = df.apply(lambda col: col[col.isnull()].size) / float(df.shape[0])
    return list(null_fracs[null_fracs >= frac].index)

def find_binary_cols(df):
    '''
    Given a dataframe, return the names of columns containing only binary values {0,1}
    '''
    binary_cols = []
    for cname in df:
        col = df[cname]
        unique_vals = col[~col.isnull()].value_counts().index
        unique_vals_len = len(unique_vals)
        # If a column contains more than 2 unique values, then it's not binary
        if unique_vals_len > 2:
            continue
        # |{0,1} ^ {0|1}| == 1
        # |{0,1} ^ {0,1}| == 2
        unique_vals_set = set(unique_vals)        
        if len(unique_vals_set.intersection({0,1})) == unique_vals_len:
            binary_cols.append(cname)
    return binary_cols

def plot_null(df, title='nulls', sort=True):
    '''
    Plot the nulls in each column of dataframe
    '''
    col_nulls = pd.isnull(df).sum()
    if sort:
        col_nulls.sort()
    plt.plot(col_nulls);
    plt.title(title)
    return col_nulls

def plot_inf(df, title='infs', sort=True):
    '''
    Plot the infs in each column of dataframe
    '''
    col_infs = np.isinf(df).sum()
    if sort:
        col_infs.sort()
    plt.plot(col_infs);
    plt.title(title)
    return col_infs

def plot_null_inf(df, sort=True):
    '''
    Plot the distribution of nulls in each column
    '''
    plt.figure(figsize=(16, 6))
    # Nulls
    plt.subplot(121)
    col_nulls = plot_null(df, sort=sort)
    # Infs
    plt.subplot(122)
    col_inf = plot_inf(df, sort=sort)
    plt.show()
    return col_nulls, col_inf
