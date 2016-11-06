"""Exploratory Data Analysis
"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2014.02.13
# Last Modified   : 2016.04.20
#
# License         : MIT

import csv
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import queue
import seaborn as sns
import sys
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from matplotlib import style
from pandas.tools.plotting import scatter_matrix
from prettytable import PrettyTable
from sklearn.decomposition import PCA

from datsci import dataio

style.use('ggplot')
sns.set_style("whitegrid")
sns.set_context(rc={
    "figure.figsize": (16, 10),
    "axes.titlesize": 14
})


def get_dummy():
    """Return a simple dataframe for testing something
    """
    return pd.DataFrame({'a': [1, 2, 3],
                         'b': [4, 5, 6],
                         'c': ['d', 'e', 'f']})


def pprint(df):
    """Pretty-print data frame
    """
    table = PrettyTable([''] + list(df.columns))
    for row in df.itertuples():
        table.add_row(row)
    print(table)


def df_isclose(df1, df2, tol=1e-8):
    """Determine if the numeric values in 2 dataframes are very close in values,
    essentially equal with tolerance tol
    """
    df1_num = df1.select_dtypes(include=[np.number])
    df2_num = df2.select_dtypes(include=[np.number])
    return np.isclose(
        df1_num.values,
        df2_num.values,
        atol=tol,
        equal_nan=True,
    ).sum() == np.prod(df1_num.shape) == np.prod(df2_num.shape)


def find_const_cols(df, dropna=True):
    """Find constant (uninformative) columns
    Returns columns with all the same values (excluding nulls)
    """
    return find_n_nary_cols(df, n=1, dropna=dropna)


def find_null_cols(df, frac=.8):
    """Find columns containing >= frac null values
    """
    def compute_frac_null(col):
        return col[col.isnull()].size / float(df.shape[0])
    null_fracs = df.apply(compute_frac_null)
    return list(null_fracs[null_fracs >= frac].index)


def find_n_nary_cols(df, n=2, dropna=True):
    """Given a dataframe, return the names of columns containing only
    n unique values. For example, binary columns contain n=2 unique values
    """
    return_cols = []
    for c in df:
        col = df[c]
        if dropna:
            col = col[~col.isnull()]
        if col.nunique(dropna=dropna) == n:
            return_cols.append(c)
    return return_cols


def get_hist_unique_col_values(df, dropna=True):
    """Get histogram of the number of unique values per column
    """
    num_unique_vals = []
    for c in df:
        col = df[c]
        if dropna:
            col = col[~col.isnull()]
        num_unique_vals.append(col.nunique(dropna=dropna))
    return pd.Series(num_unique_vals).value_counts().sort_index()


def get_hist_unique_col_values_many(dfs, columns, dropna=True):
    """Return the results of get_hist_unique_col_values on multiple DataFrames
    """
    results = {}
    for i, df in enumerate(dfs):
        results[columns[i]] = get_hist_unique_col_values(
            df, dropna=dropna)
    return pd.DataFrame(results)[columns]


def find_categorical_columns(df_train, df_test):
    """Find columns that are categorical by matching unique values between
    the train and test data
    Return list of tuples, each tuple containing column name and counts
    i.e. [('column1', 1), ('column2', 2), ...]
    """
    categorical_cols = []
    for c in df_train:
        col_train = df_train[c]
        col_train_uniq = col_train.unique(dropna=True)
        col_test = df_test[c]
        col_test_uniq = col_test.unique(dropna=True)
        if set(col_train_uniq) == set(col_test_uniq):
            categorical_cols.append((c, col_train_uniq.size))
    return sorted(categorical_cols, key=lambda p: p[1])


def find_extreme_cols(df, T=10000):
    """Return a list of column names that have extreme values that might
    be in place of NaNs, i.e. -9999999
    Checks each column to see if abs(max|min) >= T * (75th percentile)
    """
    possible_cols = []
    for c in df.columns:

        # Skip string columns
        if df[c].dtype == np.dtype('O'):
            continue

        d = df[c].describe()
        minval = abs(d['min'])
        maxval = abs(d['max'])
        thresh = abs(d['75%'] * T)
        if minval >= thresh or maxval >= thresh:
            possible_cols.append(c)
    return possible_cols


def summarize_nulls(train, test, add_info_names=[], add_info_dicts=[]):
    """Summarize null values in train and test data.
    Output contains only columns that have null values.
    """
    n_train, n_test = train.shape[0], test.shape[0]
    print('Num train samples: {}'.format(n_train))
    print('Num test samples: {}'.format(n_test))
    null_summary = []
    for c in train.columns:
        num_nan_train = train[train[c].isnull()].shape[0]
        perc_nan_train = 100.0 * num_nan_train / n_train
        try:
            num_nan_test = test[test[c].isnull()].shape[0]
        except KeyError:
            num_nan_test = n_test
        perc_nan_test = 100.0 * num_nan_test / n_test
        if num_nan_train or num_nan_test > 0:
            row = [c, num_nan_train, perc_nan_train,
                   num_nan_test, perc_nan_test]
            row += [d[c] for d in add_info_dicts]
            null_summary.append(row)
    colnames = ['Column', 'Null Train', '% Train',
                'Null Test', '% Test']
    colnames += add_info_names
    return pd.DataFrame(null_summary, columns=colnames)


def plot_null(df, title='nulls', sort=True, percent=True):
    """Plot the nulls in each column of dataframe
    """
    col_nulls = pd.isnull(df).sum()
    if percent:
        col_nulls = col_nulls / float(df.shape[0])
    if sort:
        col_nulls.sort()
    plt.plot(col_nulls);
    plt.title(title)
    return col_nulls


def plot_inf(df, title='infs', sort=True, percent=True):
    """Plot the infs in each column of dataframe
    """
    col_infs = np.isinf(df).sum()
    if percent:
        col_infs = col_infs / float(df.shape[0])
    if sort:
        col_infs.sort()
    plt.plot(col_infs)
    plt.title(title)
    return col_infs


def plot_null_inf(df, sort=True, percent=True):
    """Plot the distribution of nulls in each column
    """
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
    """Find clusters of correlated columns by first computing correlation between
    the columns and then grouping the columns based on a threshold.

    Returns a list containing sets of clustered columns

    Uses BFS to find all column clusters
    """
    df_corr = df.corr(method=method)

    # Set nodes to be the column names of the data frame
    if cols is None:
        nodes = df.columns
    else:
        nodes = cols

    def get_neighbors(n):
        """Given a node n, get all other nodes that are connected to it
        """
        neighbors = set(df_corr[df_corr[n] >= thresh].index)
        if neighbors:
            neighbors.remove(n)
        return neighbors

    def get_cluster(n):
        """Given a node n, find all connected nodes using BFS
        """
        q = queue.Queue(len(nodes))
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


def summarize_training_data(df, y_name='Label', summary_pkl='summary_data.pkl'):
    """Summarize columnar data

    Input:
      df: pandas DataFrame object containing training data
      y_name: column name of class labels or target y values
      summary_pkl: Name of output .pkl file for storing summary data.
                   Set to None in order to prevent output

    Returns tuple containing the following:
      DataFrame containing column summaries
      Number of total rows
      Number of unique labels/categories
    """
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
            pickle.dump(summary_data, f)
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
        colnames = next(reader)
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
            pickle.dump(summary_data, f)

    return df_summary, n_rows, label_counts


def load_summary_data(summary_pkl='summary_data.pkl'):
    '''
    Load summary pickle data
    '''
    with open(summary_pkl, 'rb') as f:
        summary_data = pickle.load(f)
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


def biplot(df):
    """
    Create biplot using principle components 1 and 2
    Play around with the ranges for scaling the plot
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1, 3])
    """
    # Fit on 2 components
    pca = PCA(n_components=2, whiten=True).fit(df)

    # Plot transformed/projected data
    ax = pd.DataFrame(
        pca.transform(df),
        columns=['PC1', 'PC2']
    ).plot(kind='scatter', x='PC1', y='PC2', figsize=(10, 8), s=0.8)

    # Plot arrows and labels
    pc1_pc2 = zip(pca.components_[0], pca.components_[1])
    for i, (pc1, pc2) in enumerate(pc1_pc2):
        ax.arrow(0, 0, pc1, pc2, width=0.001, fc='orange', ec='orange')
        ax.annotate(df.columns[i], (pc1, pc2), size=12)

    return ax


def corr_heat(df):
    """Plot heat map of the correlation matrix of the dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        Contains columnar data that will be used to compute
        the correlation matrix

    Returns
    -------
    corr_matrix : pandas.DataFrame
        df.corr()

    ax : matplotlib.axes._subplots.AxesSubplot
        Useful for manipulating plot
    """
    corr = df.corr()

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 9))

    # Draw the heatmap using seaborn
    ax = sns.heatmap(corr, square=True, cmap='seismic')

    return corr, ax


def most_correlated(df, thresh=0.8):
    """Find pairs of columns that are highly correlated

    Parameters
    ----------
    df : pandas.DataFrame
        Contains columnar data that will be used to compute
        the correlation matrix

    Returns
    -------
    pandas.Series
    """
    corr = df.corr()
    # Lower triangle of the correlation matrix, below the diagonal of 1's
    corr.loc[:, :] = np.tril(corr, k=-1)
    corr = corr.stack()
    return corr[(corr > thresh) | (corr < -thresh)]


def corr_bin_w_numeric(df, bin_col, numeric_col):
    """Estimate correlation between a column containing binary data and
    a column with numerical data by rank-ordering

    Parameters
    ---------
    df : pandas.DataFrame
        Contains columnar data containing the `bin_col` and `numeric_col` columns

    bin_col : str
        Name of column containing binary data

    numeric_col : str
        Name of column containing numerical data

    Returns
    -------
    corr_estimate : float
        Estimated correlation between the values of the two columns

    Reference
    ---------
    http://stackoverflow.com/questions/13269890/cartesian-product-in-pandas
    http://stats.stackexchange.com/questions/102778/correlations-between-continuous-and-categorical-nominal-variables
    """
    # Create a deep copy of the columns that we care about
    _df = df[[bin_col, numeric_col]].copy(deep=True)
    _df['dummy'] = 0  # Create dummy col for Cartesian product
    _df = pd.merge(
        _df[_df[bin_col] == 0],
        _df[_df[bin_col] == 1],
        on='dummy', suffixes=('_0', '_1'))
    n_0_gt_1 = _df[_df[numeric_col + '_0'] > _df[numeric_col + '_1']].shape[0]
    n_1_gt_0 = _df[_df[numeric_col + '_1'] > _df[numeric_col + '_0']].shape[0]
    return max(n_1_gt_0, n_0_gt_1) / float(n_0_gt_1 + n_1_gt_0)


def corrs_bin_w_numeric(df, bin_col, columns):
    """Compute estimate of correlations between every column
    in `columns` with bin_col
    """
    corrs = []
    for c in columns:
        corrs.append(corr_bin_w_numeric(df, bin_col, c))
    return pd.Series(corrs, name='corr', index=columns)


def scatter_plot_bin_vs_numeric(df, bin_col, numeric_col, figsize=(8, 6)):
    """Scatter plot a binary target column vs a numeric valued column

    Parameters
    ----------
    df : pandas.DataFrame
        Contains columnar data containing the `bin_col` and `numeric_col` columns

    bin_col : str
        Name of column containing binary data

    numeric_col : str
        Name of column containing numerical data

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object for the plot
    """
    _df = df[[bin_col, numeric_col]].copy(deep=True)
    ax = _df[_df[bin_col] == 1].plot(
        kind='scatter', x=bin_col, y=numeric_col, figsize=figsize, color='b')
    _df[_df[bin_col] == 0].plot(
        kind='scatter', x=bin_col, y=numeric_col, color='r', ax=ax)
    return ax


def scatter_matrix_bin_target(df, bin_col, numeric_cols):
    """Scatter matrix of numerical columns, showing colors based
    on a binary target variable

    Parameters
    ----------
    df : pandas.DataFrame
        Contains columnar data containing `bin_col` and `numeric_cols` columns

    bin_col : str
        Name of column containing binary data

    numeric_cols : [str]
        List containing column names containing numerical data

    Reference
    ---------
    http://stackoverflow.com/questions/28034424/pandas-scatter-matrix-plot-categorical-variables
    """
    _scatter_color = df[bin_col].apply(lambda v: ('red', 'blue')[v])
    scatter_matrix(df[numeric_cols], c=_scatter_color)
