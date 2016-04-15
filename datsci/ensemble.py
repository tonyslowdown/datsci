"""Ensembling model predictions
"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2016.04.12
# Last Modified   : 2016.04.12
#
# License         : MIT

import numpy as np
import pandas as pd


def weighted_avg(y_hats, weights=[]):
    """Computes weighted average of predicted values.

    Parameters
    ----------
    y_hats : Iterable of numpy.ndarray's
        Machine learning model-predicted values.

    weights : Iterable
        Weights corresponding to y_hats in the same order.
        If weights is empty, then the model just returns unweighted mean.

    Returns
    -------
    wavg : numpy.ndarray
        Weighted averages
    """
    df_y_hats = pd.DataFrame(y_hats).T
    if not weights:
        return df_y_hats.mean(axis=1).values

    return df_y_hats.apply(
        lambda row: np.dot(row.values, weights), axis=1).values / sum(weights)
