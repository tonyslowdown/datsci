#!/usr/bin/env python
'''
Description     : Module to handle data munging/wrangling
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2014.02.13
Last Modified   : 
Modified By     : 
'''

import numpy as np
import pandas as pd
import re

def scale_down(strval, mvleft=6):
    '''
    Scale down a string numeric value by moving the decimal to the left
    mvleft times
    '''

    # Raise error for negative mvleft values
    if mvleft < 1:
        raise ValueError('mvleft must be an integer greater than 0\n')

    # Raise error if strval is non-numeric string
    if type(strval) == str and not re.search(r'^[0-9.]+$', strval):
        raise ValueError('Could not scale non-numeric string {0}\n'.format(strval))

    # If value is None or np.nan, just return the value
    if type(strval) != str:
        if pd.isnull(strval):
            return strval
        elif np.isnan(strval):
            return strval
        else:
            raise ValueError('Could not scale value {0} of type {1}\n'.format(strval, type(strval)))

    # strval cannot have more than one decimal
    if strval.count('.') > 1:
        raise ValueError('Value contains more than one decimal {0}\n'.format(strval))

    # Split up the string value by decimal
    strvals = strval.split('.')
    strval_left = strvals[0]
    strval_right = strvals[1] if len(strvals) == 2 else '0'
    strval_left_len = len(strval_left)
    if strval_left_len <= mvleft:
        strval_left = ''.join(['0' for p in range(mvleft - strval_left_len + 1)]) + strval_left
    return float('.'.join([strval_left[:-mvleft],
                           strval_left[-mvleft:] + strval_right]))

def scale_down_cols(df, cols, mvleft=6):
    '''
    Scale down the values in several columns by moving the decimal in each value mvleft times left
    '''
    df2 = df.copy(deep=True)
    for col in cols:
        df2[col] = df2[col].apply(scale_down, mvleft=mvleft)
    return df2
