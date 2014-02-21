#!/usr/bin/env python
'''
Description     : Module to handle EDA (Exploratory Data Analysis)
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2014.02.13
Last Modified   : 2014.02.21
Modified By     : Jin Kim jjinking(at)gmail(dot)com
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Queue
from mpltools import style; style.use('ggplot')
from sklearn.ensemble import RandomForestClassifier

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

def plot_null(df, title='nulls', sort=True, percent=True):
    '''
    Plot the nulls in each column of dataframe
    '''
    col_nulls = pd.isnull(df).sum()
    if percent:
        col_nulls = col_nulls / float(df.shape[0])
    if sort:
        col_nulls.sort()
    plt.plot(col_nulls);
    plt.title(title)
    return col_nulls

def plot_inf(df, title='infs', sort=True, percent=True):
    '''
    Plot the infs in each column of dataframe
    '''
    col_infs = np.isinf(df).sum()
    if percent:
        col_infs = col_infs / float(df.shape[0])
    if sort:
        col_infs.sort()
    plt.plot(col_infs);
    plt.title(title)
    return col_infs

def plot_null_inf(df, sort=True, percent=True):
    '''
    Plot the distribution of nulls in each column
    '''
    plt.figure(figsize=(16, 6))
    # Nulls
    plt.subplot(121)
    col_nulls = plot_null(df, sort=sort, percent=percent)
    # Infs
    plt.subplot(122)
    col_inf = plot_inf(df, sort=sort, percent=percent)
    plt.show()
    return col_nulls, col_inf

def get_column_clusters(df, cols=None, thresh=0.95, method='pearson'):
    '''
    Find clusters of correlated columns by first computing correlation between the columns
    and then grouping the columns based on a threshold

    Returns a list containing sets of clustered columns

    Uses BFS to find all column clusters
    '''
    df_corr = df.corr(method=method)

    # Set nodes to be the column names of the data frame
    if cols is None:
        nodes = df.columns
    else:
        nodes = cols

    def get_neighbors(n):
        '''
        Given a node n, get all other nodes that are connected to it
        '''
        neighbors = set(df_corr[df_corr[n] >= thresh].index)
        if neighbors:
            neighbors.remove(n)
        return neighbors

    def get_cluster(n):
        '''
        Given a node n, find all connected nodes
        Uses BFS
        '''
        q = Queue.Queue(len(nodes))
        q.put(n)
        seen = set()
        seen.add(n)
        while not q.empty():
            _n = q.get()
            for _n2 in get_neighbors(_n):
                if _n2 not in seen:
                    q.put(_n2)
                    seen.add(_n2)
        return seen

    # Iterate through every node, and create clusters based on connectivity
    clusters = []
    for cn in nodes:
        if cn not in [n for cl in clusters for n in cl]:
            clusters.append(get_cluster(cn))
    return clusters

def rank_order_features(X, y, plot=True):
    '''
    Rank order features based on their importance based on a random forest classifer

    Taken from DataGotham 2013 Data Science Tutorial, Feature Engineering
    http://nbviewer.ipython.org/urls/raw2.github.com/yhat/DataGotham2013/master/notebooks/7%20-%20Feature%20Engineering.ipynb?create=1
    '''    
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X, y);
    importances = clf.feature_importances_
    # For plotting, sort importances in increasing order
    sorted_idx = np.argsort(importances)
    colnames_sorted = X.columns[sorted_idx]
    importances_sorted = importances[sorted_idx]
    if plot:
        padding = np.arange(X.shape[1]) + 0.5
        plt.barh(padding, importances_sorted, align='center')
        plt.yticks(padding, colnames_sorted)
        plt.xlabel("Relative Importance")
        plt.title("Variable Importance")
        plt.show()
    # For return values, return the importances in decreasing order
    return colnames_sorted[::-1], importances_sorted[::-1]
