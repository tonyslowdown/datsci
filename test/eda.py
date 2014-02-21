#!/usr/bin/env python
'''
Description     : Unit test for eda.py
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2014.02.13
Last Modified   : 2014.02.21
Modified By     : Jin Kim jjinking(at)gmail(dot)com
'''

import os,sys
# Add project root to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))
import numpy as np
import pandas as pd
import scipy.stats
import unittest
from datsci import eda

class TestEda(unittest.TestCase):
    '''
    Unit tests for the eda module
    '''

    def test_find_uinfo_cols(self):
        '''
        Test finding uninformative columns in a dataframe
        '''
        df = pd.DataFrame([[1, 2, 33, 4],
                           [None, None, np.nan, 4],
                           [None, np.nan, np.nan, 44],
                           [None, 2, 33, 44],
                           [None, np.nan, 33, 44],
                           [None, None, 33, 44]], columns=['a','b','c','d'])
        self.assertEqual(eda.find_uninfo_cols(df), ['a','b','c'])

    def test_find_null_cols(self):
        '''
        Test finding columns with null values at least a certain fraction of the row
        '''
        df = pd.DataFrame([[None, 1,    1,  1.0,  1.0,    1],
                           [None, 1, None,    0,  1.0,    0],
                           [None, 2, None, None,  1.0,    0],
                           [None, 2, None, None, None,    0],
                           [None, 2, None, None, None, None]],
                          columns=['a','b','c','d','e','f'])
        self.assertEqual(eda.find_null_cols(df, frac=.99), ['a'])
        self.assertEqual(eda.find_null_cols(df, frac=.81), ['a'])
        self.assertEqual(eda.find_null_cols(df, frac=.80), ['a','c'])
        self.assertEqual(eda.find_null_cols(df, frac=.79), ['a','c'])
        self.assertEqual(eda.find_null_cols(df, frac=.60), ['a','c','d'])
        self.assertEqual(eda.find_null_cols(df, frac=.39), ['a','c','d','e'])
        self.assertEqual(eda.find_null_cols(df, frac=.20), ['a','c','d','e','f'])
        self.assertEqual(eda.find_null_cols(df, frac=0.0), ['a','b','c','d','e','f'])

    def test_find_binary_cols(self):
        '''
        Test finding binary-valued columns in a dataframe
        '''
        df = pd.DataFrame([[1, 11, 1,    1.0,  1.0, 1.000001],
                           [0, 11, None, 0,    1.0, 0],
                           [1, 22, None, None, 1.0, 0]],
                          columns=['a','b','c','d','e','f'])
        self.assertEqual(eda.find_binary_cols(df), ['a','c','d','e'])
        
    def test_get_column_clusters(self):
        '''
        Test finding clusters of correlated columns
        '''
        # Test clustering 100% correlated values
        df = pd.DataFrame([[1, 1, 1, 1, 2, 1],
                           [2, 1, 2, 2, 2, 1],
                           [3, 5, 3, 3, 1, 5],
                           [4, 5, 4, 4, 2, 5],
                           [5, 3, 5, 5, 2, 3]],
                          columns=['a','b','c','d','e','f'])
        clusts = sorted([sorted(clust) for clust in eda.get_column_clusters(df, thresh=1.0)])
        self.assertEqual(clusts, [['a','c','d'],['b','f'],['e']])

        # Test thresholding
        df = pd.DataFrame([[1, 1, 1, 1, 2, 1],
                           [2, 1, 2, 1, 2, 1],
                           [3, 5, 2, 3, 1, 5],
                           [4, 5, 4, 4, 2, 5],
                           [5, 3, 5, 5, 2, 3]],
                          columns=['a','b','c','d','e','f'])

        # Check the correlation range
        self.assertTrue(0.95 < scipy.stats.pearsonr(df.a, df.c)[0] < 0.97)

        clusts = sorted([sorted(clust) for clust in eda.get_column_clusters(df, thresh=0.95)])
        self.assertEqual(clusts, [['a','c','d'],['b','f'],['e']])

        clusts = sorted([sorted(clust) for clust in eda.get_column_clusters(df, thresh=0.97)])
        self.assertEqual(clusts, [['a','d'],['b','f'],['c'],['e']])

    def test_rank_order_features(self):
        '''
        Test rank ordering features based on importance
        '''
        df = pd.DataFrame([[0, 0, 0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 0, 1, 0],
                           [0, 0, 1, 0, 1, 0, 0],
                           [0, 1, 1, 1, 0, 1, 1],
                           [1, 0, 1, 1, 1, 0, 1],
                           [1, 1, 1, 1, 0, 1, 1]],
                          columns=['a','b','c','d','e','f','g'])
        cols_ranked, importances = eda.rank_order_features(df[['a','b','c','d','e']],
                                                           df['g'],
                                                           plot=False)
        # Check that the lengths of the two outputs are equal
        self.assertEqual(len(cols_ranked), len(importances))
        # Check that the importances are sorted
        self.assertEqual(sorted(importances, reverse=True), list(importances))
        # Check top three features
        self.assertEqual(set(cols_ranked[:3]), set(['d','c','a']))
        # Check the top feature is the same as the y
        self.assertEqual(cols_ranked[0], 'd')
        

if __name__ == '__main__':
    unittest.main()
