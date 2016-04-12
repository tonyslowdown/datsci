"""Utilities for Kaggle competitions

"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2015.12.31
# Last Modified   : 2016.04.12
#
# License         : MIT

import pandas as pd


def save_submission(y_hat, save_file,
                    sample_submission_file='data/sample_submission.csv'):
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
