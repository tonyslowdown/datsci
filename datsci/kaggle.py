'''
Description     : Utilities for kaggle
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2015.12.31
Last Modified   : 2015.12.31
Modified By     : Jin Kim jjinking(at)gmail(dot)com
'''


def save_submission(predictions, filename):
    import pandas as pd
    sample_submission = pd.read_csv('./sample_submission.csv')
    sample_submission['Response'] = predictions
    sample_submission.to_csv(filename, index=False)
