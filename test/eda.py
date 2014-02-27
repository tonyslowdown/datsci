#!/usr/bin/env python
'''
Description     : Unit test for eda.py
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2014.02.13
Last Modified   : 2014.02.25
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
from sklearn.ensemble import RandomForestClassifier

class TestEda(unittest.TestCase):
    '''
    Unit tests for the eda module
    '''

    def test_df_equal(self):
        '''
        Test checking to see if two dataframes are equal
        '''
        # Test integers
        df1 = pd.DataFrame([[1,2],
                            [3,4]])
        df2 = pd.DataFrame([[1,2],
                            [3,4]])
        self.assertTrue(eda.df_equal(df1, df2))

        df1 = pd.DataFrame([[1,2],
                            [3,5]])
        df2 = pd.DataFrame([[1,2],
                            [3,4]])
        self.assertFalse(eda.df_equal(df1, df2))

        # Test strings
        df1 = pd.DataFrame([['a',2],
                            [3,'c']])
        df2 = pd.DataFrame([['a',2],
                            [3,'c']])
        self.assertTrue(eda.df_equal(df1, df2))

        df1 = pd.DataFrame([['a',2],
                            [3,'c']])
        df2 = pd.DataFrame([['c',2],
                            [3,'c']])
        self.assertFalse(eda.df_equal(df1, df2))
        
        # Test rounding
        df1 = pd.DataFrame([[1.1234, 2.1234],
                            [3.1234, 4.1234]])
        df2 = pd.DataFrame([[1.1234, 2.1234],
                            [3.1234, 4.1234]])
        self.assertTrue(eda.df_equal(df1, df2))

        df1 = pd.DataFrame([[1.1234, 2.1234],
                            [3.1234, 5.1234]])
        df2 = pd.DataFrame([[1.1234, 2.1234],
                            [3.1234, 4.1232]])
        self.assertFalse(eda.df_equal(df1, df2))

        df1 = pd.DataFrame([[1.1234, 2.1234],
                            [3.1234, 4.1234]])
        df2 = pd.DataFrame([[1.1234, 2.1234],
                            [3.1234, 4.1232]])
        self.assertTrue(eda.df_equal(df1, df2, decimals=3))

        df1 = pd.DataFrame([[1.1234, 2.1234],
                            [3.1234, 5.1234123]])
        df2 = pd.DataFrame([[1.1234, 2.1234],
                            [3.1234, 5.123412]])
        self.assertTrue(eda.df_equal(df1, df2, decimals=6))

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
        
    def test_get_feature_clusters(self):
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
        clusts = sorted([sorted(clust) for clust in eda.get_feature_clusters(df, cols=df.columns, thresh=1.0)])
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

        clusts = sorted([sorted(clust) for clust in eda.get_feature_clusters(df, thresh=0.95)])
        self.assertEqual(clusts, [['a','c','d'],['b','f'],['e']])

        clusts = sorted([sorted(clust) for clust in eda.get_feature_clusters(df, cols=df.columns, thresh=0.97)])
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
        
    def test_generate_important_features(self):
        '''
        Test generating ordered column names uniquely from their grouped clusters
        '''
        col_clusts = [set(['a','b','c']),
                      set(['d','e','f']),
                      set(['g','h','i'])]
        gen = eda.generate_important_features(col_clusts,
                                          ['a','b','c','d','e','f','g','h','i'])
        cols = [c for c in gen]
        self.assertEqual(len(cols), 3)
        self.assertEqual(cols, ['a','d','g'])

        gen = eda.generate_important_features(col_clusts,
                                          ['b','c','a','f','e','d','h','g','i'])
        cols = [c for c in gen]
        self.assertEqual(len(cols), 3)
        self.assertEqual(cols, ['b','f','h'])

        gen = eda.generate_important_features(col_clusts,
                                          ['b','f','a','e','h','d','c','g','i'])
        cols = list(gen)
        self.assertEqual(len(cols), 3)
        self.assertEqual(cols, ['b','f','h'])

    def test_cross_validate_feature_groups(self):
        '''
        Test cross validating multiple feature groups
        '''
        # Must raise error if the number of feature groups is not equal to the number of titles
        self.assertRaises(ValueError,
                          eda.cross_validate_feature_groups,
                          'clf',
                          'df',
                          [1,2],
                          'y',
                          [1,2,3])

        df = pd.DataFrame([[0, 0, 0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 0, 1, 0],
                           [0, 0, 1, 0, 1, 0, 0],
                           [0, 1, 1, 1, 0, 1, 1],
                           [1, 0, 1, 1, 1, 0, 1],
                           [1, 1, 1, 1, 0, 1, 1]],
                          columns=['a','b','c','d','e','f','g'])
        clf = RandomForestClassifier(n_estimators=10)
        cv_results = eda.cross_validate_feature_groups(clf, df, [['a','b'],['c','d'],['e','f']],
                                                       df.g, cv=3,
                                                       titles=['group 1(a,b)', 'group 2(c,d)',
                                                               'group 3(c,d)'], plot=False)
        self.assertEqual(cv_results.shape, (3,3))


if __name__ == '__main__':
    unittest.main()
