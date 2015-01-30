#!/usr/bin/env python
'''
Description     : Module to handle EDA (Exploratory Data Analysis)
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2014.02.13
Last Modified   : 2015.01.30
Modified By     : Jin Kim jjinking(at)gmail(dot)com
'''

import csv
import cPickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Queue
import sys
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from mpltools import style; style.use('ggplot')
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

import dataio

def df_equal(df1, df2, decimals=None):
    '''
    Compare the values of two pandas DataFrame objects element by element,
    and if every single element is equal, return True

    Parameter decimals determines the number of decimal places to round decimal
    values before comparing
    '''
    # First, compare the sizes
    if df1.shape != df2.shape:
        return False

    # Compare values, and round decimals
    n_elements = np.multiply(*df1.shape)
    l1 = np.squeeze(df1.values.reshape(n_elements, 1))
    l2 = np.squeeze(df2.values.reshape(n_elements, 1))
    if decimals is not None and isinstance(decimals, int):
        l1 = np.round(l1, decimals=decimals)
        l2 = np.round(l2, decimals=decimals)
    for t in range(len(l1)):
        a,b = l1[t], l2[t]
        # If both are np.nan, skip
        if not isinstance(a, str) and not isinstance(b, str):
            if np.isnan(a) and np.isnan(b):
                continue
        # Regular comparison
        if a != b:
            return False
    return True

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

def get_feature_clusters(df, cols=None, thresh=0.95, method='pearson'):
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

def rank_order_features(X, y, clf=None, plot=True):
    '''
    Rank order features based on their importance based on a random forest classifer

    Taken from DataGotham 2013 Data Science Tutorial, Feature Engineering
    http://nbviewer.ipython.org/urls/raw2.github.com/yhat/DataGotham2013/master/notebooks/7%20-%20Feature%20Engineering.ipynb?create=1
    '''
    if clf is None:
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

def generate_important_features(col_clusts, col_order):
    '''
    Given a list of clusters containing column names and a list of ordered column names,
    generate important column names one by one, skipping columns who belong to clusters
    where a member of the cluster has already been returned.

    col_clusts is in the format [set(), set(), set()] where each set represents a column
    col_order is an ordered collection containing ordered column names.
    '''
    # Create a dictionary of column names to the index of the cluster they belong to
    col2clustname = {}
    for i,clust in enumerate(col_clusts):
       for col in clust:
           col2clustname[col] = i

    # Maintain a set of cluster names where a member of the cluster has already been returned
    clusts_visited = set()
    for c in col_order:
        if col2clustname[c] not in clusts_visited:
            clusts_visited.add(col2clustname[c])
            yield c

def cross_validate_feature_groups(clf, df, feature_groups, y, titles=None, 
                                  cv=10, plot=True, plots_per_row=4):
    '''
    Given a classifier and a list of feature groups, run cross-validation on the feature groups
    and generate plots on the scores, and return a dataframe of score summaries.
    '''
    # Ensure that the length of titles is equal to the length of feature groups
    len_feature_groups = len(feature_groups)
    if (titles is not None) and (len(titles) != len_feature_groups):
        raise ValueError('Length of titles must be equal to length of feature groups')

    # If titles are not set, then set numeric titles
    if titles is None:
        titles = range(len_feature_groups)

    # If plot is turned on
    if plot:
        fig = plt.figure(figsize=(16,10))
        n_rows = np.ceil(len_feature_groups / float(plots_per_row))

    ax0 = None
    g2scores = {}
    for i,g in enumerate(feature_groups):
        scores = cross_validation.cross_val_score(clf, df[g], y, cv=cv)
        g2scores[titles[i]] = {'mean': scores.mean(),
                               'max': scores.max(),
                               'min': scores.min()}
        # If plot is turned on
        if plot:
            if i == 0:
                ax0 = plt.subplot(n_rows, plots_per_row, i + 1)
            else:
                plt.subplot(n_rows, plots_per_row, i + 1, sharey=ax0)
            plt.plot(scores);
            plt.title(titles[i])
    
    # Show plot
    if plot:
        fig.tight_layout()

    # Return scores summary
    scores_summary = pd.DataFrame(g2scores)[titles]

    # Bar graph of the scores summary
    if plot:
        scores_summary.transpose().plot(kind='bar', figsize=(16,8))
        plt.title('Scores Summary')
        plt.legend(loc=4)

    return scores_summary

def summarize_training_data(df, y_name='Label', summary_pkl='summary_data.pkl'):
    '''
    Summarize columnar data

    Input:
      df: pandas DataFrame object containing training data
      y_name: column name of class labels or target y values
      summary_pkl: Name of output .pkl file for storing summary data.
                   Set to None in order to prevent output
      
    Returns tuple containing the following:
      DataFrame containing column summaries
      Number of total rows
      Number of unique labels/categories
    '''
    def _is_nan(val):
        '''
        Runs np.isnan on a value if it's float type
        '''
        if isinstance(val, float):
            return np.isnan(val)
        return False

    def _is_null_or_blank(val):
        '''
        Check to see if value is null or blank string
        '''
        # If numeric type, and is zero, return False
        if isinstance(val, int) or isinstance(val, float):
            if val == 0:
                return False
        return not val or pd.isnull(val)

    def _get_uniq(series):
        '''
        Get number of unique items in series
        '''
        s = set(series.values)
        null_exists = False
        n_unique = 0
        for val in s:
            if _is_null_or_blank(val):
                null_exists = True
                continue
            n_unique += 1
        if null_exists:
            n_unique += 1
        return n_unique

    def _get_min_max(series):
        '''
        Get maximum value in a pandas Series
        '''
        minval = np.inf
        maxval = -np.inf
        for val in series:
            # Skip empty or null values
            if _is_null_or_blank(val):
                continue
            try:
                val = float(val)
                minval = min(minval, val)
                maxval = max(maxval, val)
            except ValueError:
                return np.nan, np.nan
        if np.isinf(minval):
            minval = np.nan
        if np.isinf(maxval):
            maxval = np.nan
        return minval, maxval

    summary_data = defaultdict(list)
    n_rows = float(df.shape[0])
    for colname in df.columns:
        summary_data['attribute'].append(colname)
        minval, maxval = _get_min_max(df[colname])
        summary_data['max'].append(maxval)
        summary_data['min'].append(minval)
        summary_data['n_null'].append(df[colname].apply(_is_null_or_blank).sum())
        # Counting n_uniq can be thrown off by np.nan columns, which are not able to be dedupped
        # Therefore, must count number of np.nans in the column, and subtract appropriately
        #n_uniq = df[colname].drop_duplicates().shape[0]
        #n_null = df[colname].apply(_is_nan).sum()
        #if n_null > 1:
        #    n_uniq = n_uniq - n_null + 1
        summary_data['n_uniq'].append(_get_uniq(df[colname]))

    df_summary = pd.DataFrame(summary_data)
    df_summary['perc_null'] = df_summary['n_null'] / n_rows
    label_counts = df[y_name].value_counts(dropna=False).to_dict()

    if summary_pkl is not None:
        summary_data = {'summary': df_summary,
                        'n_rows': n_rows,
                        'label_counts': label_counts}
        with open(summary_pkl, 'wb') as f:
            cPickle.dump(summary_data, f)
    
    return df_summary, n_rows, label_counts

def summarize_big_training_data(fname,
                                y_name='Label',
                                n_uniq_toomany=1000,
                                progress_int=None,
                                summary_pkl='summary_data.pkl'):
    '''
    Summarize columnar data
    
    Input:
      fname: input file name
      y_name: column name of class labels or target y values
      n_uniq_toomany: number of unique column values considered too many to 
                      continue counting
      progress_int: Output progress every progress_int number of rows of input
      summary_pkl: Name of output .pkl file for storing summary data.
                   Set to None in order to prevent output

    Returns tuple containing the following:
      DataFrame containing column summaries
      Number of total rows
      Dictionary containing y(label) value counts
    '''
    # Number of rows total
    n_rows = 0
    # Total number of instances for each class label
    label_counts = defaultdict(int)
    # Total number of null values per column
    null_counts = defaultdict(int)
    # Max and min values per column
    col_max = defaultdict(lambda: -np.inf)
    col_min = defaultdict(lambda: np.inf)
    col_numeric = defaultdict(lambda: True)
    # Number of unique values
    col_uniq_vals = defaultdict(set)
    col_uniq_vals_toomany = set()

    with closing(dataio.fopen(fname)) as fin:
        reader = csv.reader(fin)
        # Store colnames
        colnames = reader.next()
        for t,row in enumerate(reader):
            # Output progress
            if progress_int is not None and t % progress_int == 0:
                sys.stdout.write('{}\tencountered: {}\n'.format(datetime.now(), t))

            # Increment count of rows
            n_rows += 1
            
            # Create dictionary mapping colnames to each row value
            row_dict = dict(zip(colnames, row))
            
            # Update label couts
            if y_name not in col_uniq_vals_toomany:
                label_counts[row_dict[y_name]] += 1

            # Loop through cols
            for colname in colnames:
                
                # Update null counts
                col_val = row_dict[colname].strip()
                if not col_val:
                    null_counts[colname] += 1
                    
                # Update max and min values
                if col_val and col_numeric[colname]:
                    try:
                        col_val = float(col_val)
                        col_max[colname] = max(col_max[colname],
                                               col_val)
                        col_min[colname] = min(col_min[colname],
                                               col_val)
                    except ValueError:
                        col_numeric[colname] = False
                        
                # Update unique values per column
                uniq_vals_thiscol = col_uniq_vals[colname]
                if colname not in col_uniq_vals_toomany:
                    uniq_vals_thiscol.add(col_val)
                if len(uniq_vals_thiscol) > n_uniq_toomany:
                    col_uniq_vals_toomany.add(colname)

    summary_data = defaultdict(list)
    for colname in colnames:
        summary_data['attribute'].append(colname)
        summary_data['n_null'].append(null_counts[colname])
        summary_data['perc_null'].append(float(null_counts[colname])/n_rows)
        colmax, colmin = None, None
        if col_numeric[colname]:
            colmax = col_max[colname] if not np.isinf(col_max[colname]) else None
            colmin = col_min[colname] if not np.isinf(col_min[colname]) else None
        summary_data['max'].append(colmax)
        summary_data['min'].append(colmin)
            
        # Count number of unique values
        if colname in col_uniq_vals_toomany:
            n_uniq = '> {}'.format(n_uniq_toomany)
        else:
            n_uniq = len(col_uniq_vals[colname]) 
        summary_data['n_uniq'].append(n_uniq)

    # If there are too many y-values, set label_counts to None
    if y_name in col_uniq_vals_toomany:
        label_counts = None

    df_summary = pd.DataFrame(summary_data)

    if summary_pkl is not None:
        summary_data = {'summary': df_summary,
                        'n_rows': n_rows,
                        'label_counts': label_counts}
        with open(summary_pkl, 'wb') as f:
            cPickle.dump(summary_data, f)

    return df_summary, n_rows, label_counts

def load_summary_data(summary_pkl='summary_data.pkl'):
    '''
    Load summary pickle data
    '''
    with open(summary_pkl, 'rb') as f:
        summary_data = cPickle.load(f)
    return (summary_data['summary'],
            summary_data['n_rows'],
            summary_data['label_counts'])

def count_big_file_value_counts(fname, colname):
    '''
    Count the number of occurrances for each unique value in a column
    Returns a defaultdict containing the value counts
    '''
    value_counts = defaultdict(int)
    with closing(dataio.fopen(fname)) as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            value_counts[row[colname]] += 1
    return value_counts
