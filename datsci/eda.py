#!/usr/bin/env python
'''
Description     : Module to handle EDA (Exploratory Data Analysis)
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2014.02.13
Last Modified   : 
Modified By     : 
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

def plot_null(df, title='nulls'):
    '''
    Plot the nulls in each column of dataframe
    '''
    col_nulls = pd.isnull(df).sum()
    col_nulls.sort()
    plt.plot(col_nulls);
    plt.title(title)

def plot_inf(df, title='infs'):
    '''
    Plot the infs in each column of dataframe
    '''
    col_infs = np.isinf(df).sum()
    col_infs.sort()
    plt.plot(col_infs);
    plt.title(title)

def plot_null_inf(df):
    '''
    Plot the distribution of nulls in each column
    '''
    plt.figure(figsize=(16, 6))
    # Nulls
    plt.subplot(121)
    plot_null(df)
    # Infs
    plt.subplot(122)
    plot_inf(df)
    plt.show()
