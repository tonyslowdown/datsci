"""Handle io

"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2013.09.23
# Last Modified   : 2016.04.15
#
# License         : MIT

import csv
import os
import pandas as pd
import random
import sys
from datetime import datetime


def fopen(fname, mode='r'):
    """
    Given fname, a full file path as a string, return correct file handle
    based on extension
    File mode is either 'r' or 'w'
    For compressed formats, assumes there is only a single file within
    an archive
    Can handle the following: txt, csv, tsv, zip, gz, tar.gz, tar.bz2
    Usages may be different for various compression libraries
    """
    import gzip
    import tarfile
    import zipfile

    # Process file type
    if not isinstance(fname, str):
        raise TypeError('File name must be string')

    fbase, fext = os.path.splitext(fname)
    fext2 = os.path.splitext(fbase)[1]  # second to last extension

    # zip
    if fext == '.zip':
        if mode == 'w':
            raise NotImplementedError(
                'Cannot write to given format: {}\n'.format(fext))

        f = zipfile.ZipFile(fname, "r")
        return f.open(f.namelist()[0])

    # tar.gz and tar.bz2
    elif fext2 == '.tar':
        if mode == 'w':
            raise NotImplementedError(
                'Cannot write to given format: {}\n'.format(fext2 + fext))

        # tar.gz file
        if fext == '.gz':
            _mode = 'r:gz'
        # tar.bz2 file
        elif fext == '.bz2':
            _mode = 'r:bz2'
        f = tarfile.open(fname, _mode)
        return f.extractfile(f.getnames()[0])

    # Just gz file
    elif fext == '.gz':
        return gzip.open(fname, mode + 'b')

    # All other formats
    return open(fname, mode)


def reservoir_sample(iterable, k, rseed=None, progress_int=None):
    '''
    Select k random elements from an iterable containing n elements
    into a list with equal probability, where n is large and unknown.

    Allows repeable results with random seed rseed.

    Uses the reservoir sampling algorithm taken from the wikipedia article
    (http://en.wikipedia.org/wiki/Reservoir_sampling)
    '''
    random.seed(rseed)
    selected_samples = []
    for i, el in enumerate(iterable):
        # Output progress
        if progress_int is not None and i % progress_int == 0:
            sys.stdout.write('{}\tencountered: {}\n'.format(datetime.now(), i))

        # Fill up reservoir
        if i < k:
            selected_samples.append(el)
        else:
            # Choose an integer in [0,i]
            j = random.randint(0, i)
            if j < k:
                selected_samples[j] = el
    return selected_samples


def load_subset(fname, k=None, sep=',', colnames=None,
                header=0, rseed=None, progress_int=None):
    '''
    Load k rows from file into a dataframe from file fname
    Setting colnames to a list will set that as the column
    name of the returned DataFrame object
    Setting header to None will result in reading the first
    row into the data frame as data instead of header
    '''
    # Read in entire file
    if k is None:
        return pd.read_csv(fname, sep=sep, names=colnames, header=header)

    # Subsample from file
    with fopen(fname) as fin:
        reader = csv.reader(fin, delimiter=sep)

        # Handle column names logic
        names = None
        if header == 0:
            names = reader.next()
        if colnames is not None:
            names = colnames

        return pd.DataFrame(reservoir_sample(
            reader, k, rseed=rseed, progress_int=progress_int), columns=names)


def head(fname, k=10, sep=','):
    '''
    Load first k lines of a file
    '''
    with fopen(fname) as fin:
        reader = csv.reader(fin, delimiter=sep)
        colnames = reader.next()
        if k == 0:
            return pd.Series(colnames)

        data = []
        n = 0
        for row in reader:
            if n >= k:
                break
            data.append(row)
            n += 1
    return pd.DataFrame(data, columns=colnames)
