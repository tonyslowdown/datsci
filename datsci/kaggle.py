'''
Description     : Utilities for kaggle
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2015.12.31
Last Modified   : 2016.03.16
Modified By     : Jin Kim jjinking(at)gmail(dot)com
'''

import pandas as pd


def save_submission(predictions, save_file,
                    sample_submission_file='data/sample_submission.csv'):
    '''
    Format predicted values for test set, and save
    for submission to kaggle
    '''
    df = pd.read_csv(sample_submission_file)
    df[df.columns[1]] = predictions
    df.to_csv(save_file, index=False)
