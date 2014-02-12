#!/usr/bin/env python
'''
Description     : Module to handle data files
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2013.09.23
Last Modified   : 2014.02.12
Modified By     : Jin Kim jjinking(at)gmail(dot)com
'''

import csv
import pandas as pd
import random

def reservoir_sample(iterable, k, rseed=None):
    '''
    Select k random elements from an iterable containing n elements
    into a list with equal probability, where n is large and unknown.
    Enable repeable results with random seed rseed.
    Uses the reservoir sampling algorithm taken from the wikipedia article
    (http://en.wikipedia.org/wiki/Reservoir_sampling)
    '''
    random.seed(rseed)
    selected_samples = []
    for i,el in enumerate(iterable):
        # Fill up reservoir
        if i < k:
            selected_samples.append(el)
        else:
            # Choose an integer in [0,i]
            j = random.randint(0, i)
            if j < k:
                selected_samples[j] = el
    return selected_samples

def load_subset(f, k=None, sep=',', colnames=None, header=0, rseed=None):
    '''
    Load k rows from file into a dataframe from file of buffer f
    Setting colnames to a list will set that as the column name of the returned DataFrame object
    Setting header to None will result in reading the entire file into the data frame as data,
    not treating the first row of the file as column header
    '''
    # Read in entire file
    if k is None:
        return pd.read_csv(f, sep=sep, names=colnames, header=header)
    
    # Subsample from file
    if type(f) == str:
        f = open(f, 'rU')
    reader = csv.reader(f, delimiter=sep)
    
    # Handle column names logic
    names = None
    if header == 0:
        names = reader.next()
    if colnames is not None:
        names = colnames

    return pd.DataFrame(reservoir_sample(reader, k, rseed=rseed),
                        columns=names)



