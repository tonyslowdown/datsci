"""Utilities for Kaggle competitions

"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2015.12.31
# Last Modified   : 2016.04.12
#
# License         : MIT

import pandas as pd

from datsci import ensemble


DEFAULT_SAMPLE_SUBMISSION_FILE = 'data/sample_submission.csv'
DEFAULT_SAMPLE_SUBMISSION_IDX = 'ID'


def save_submission(y_hat, save_file,
                    sample_submission_file=DEFAULT_SAMPLE_SUBMISSION_FILE):
    """Format predicted values for test set, and save for submission to kaggle

    Parameters
    ----------
    y_hat : numpy.ndarray or pandas.Series
        Machine learning model-predicted values.

    save_file : str
        File name to save the predicted values, including path

    sample_submission_file : str
        Path to example submission file provided by Kaggle
    """
    df = pd.read_csv(sample_submission_file)
    df[df.columns[1]] = y_hat
    df.to_csv(save_file, index=False)


def weighted_avg_from_files(
        fnames, outfile, weights=[],
        sample_submission_file=DEFAULT_SAMPLE_SUBMISSION_FILE,
        sample_submission_idx=DEFAULT_SAMPLE_SUBMISSION_IDX):
    """Compute weighted avg from submission files, and save results to a new file

    Parameters
    ----------
    fnames : Iterable of str's
        Submission file names of y_hats to be averaged

    outfile : str
        Output file name, including path

    weights : Iterable
        Weights corresponding to y_hats in the same order.
        If weights is empty, then the model just returns unweighted mean.

    sample_submission_file : str
        Path to example submission file provided by Kaggle

    sample_submission_idx : str
        Index column name in `sample_submission_file`

    Returns
    -------
    y_hat_avg : numpy.ndarray
        Weighted averages
    """
    y_hat_avg = ensemble.weighted_avg(
        [pd.read_csv(f, index_col=sample_submission_idx, squeeze=True).values
         for f in fnames],
        weights=weights
    )
    save_submission(y_hat_avg, outfile,
                    sample_submission_file=sample_submission_file)
    return y_hat_avg
