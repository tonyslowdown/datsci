#!/usr/bin/env python
'''
Description     : Unit test for munge.py
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2014.02.13
Last Modified   : 2014.09.21
Modified By     : Jin Kim jjinking(at)gmail(dot)com
'''

import os,sys
# Add project root to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))
import numpy as np
import pandas as pd
import unittest
from datsci import eda, munge

class TestMunge(unittest.TestCase):
    '''
    Unit tests for the munge module
    '''

    def setUp(self):
        '''
        Set up test environment variables
        '''
        self.curdir = os.path.dirname(os.path.abspath(__file__))

    def test_standardize_cols(self):
        '''
        Test standardizing df columns
        '''
        # Standardize all columns
        df = pd.DataFrame([[1,200,400],
                           [0,100,400],
                           [1,200,500]])
        df_std = munge.standardize_cols(df, cols=[0,1,2], ignore_binary=False)
        std_vals = np.array([[ 0.70710678,  0.70710678, -0.70710678],
                             [-1.41421356, -1.41421356, -0.70710678],
                             [ 0.70710678,  0.70710678,  1.41421356]])
        a1 = np.round(np.squeeze(df_std.values.reshape(np.multiply(*df_std.shape),1)), decimals=4)
        a2 = np.round(np.squeeze(std_vals.reshape(9,1)), decimals=4)
        self.assertTrue((a1 == a2).all())

        # Standardize some of the columns only
        df_std2 = munge.standardize_cols(df, cols=[0, 2], ignore_binary=False)
        a1 = np.round(df_std2.values[:,[0,2]].reshape(6,1), decimals=4)
        a2 = np.round(std_vals[:,[0,2]].reshape(6,1), decimals=4)
        self.assertTrue((a1 == a2).all())
        a1 = np.round(df_std2.values[:,1], decimals=4)
        a2 = np.round(std_vals[:,1], decimals=4)
        self.assertFalse((a1 == a2).any())

        # Ignore binary values
        df_std3 = munge.standardize_cols(df)
        a1 = np.round(df_std3.values[:,1:].reshape(6,1), decimals=4)
        a2 = np.round(std_vals[:,1:].reshape(6,1), decimals=4)
        self.assertTrue((a1 == a2).all())
        a1 = np.round(df_std3.values[:,0], decimals=4)
        a2 = np.round(df.values[:,0], decimals=4)
        self.assertTrue((a1 == a2).all())

    def test_impute_standardize(self):
        '''
        Test piplined method to impute and standardize training data
        '''
        df = pd.DataFrame([[1,2,3],
                           [1,2,3],
                           [None,2,3]], columns=['a','b','c'])
        self.assertTrue(eda.df_equal(munge.impute_standardize(df),
                                     pd.DataFrame([[1,0,0],
                                                   [1,0,0],
                                                   [1,0,0]])))

    def test_match_binary_labels(self):
        '''
        Test matching the number of rows based on the two y labels
        '''
        df = pd.DataFrame([[1, 0, 10.0,-1],
                           [0, 1, 20.0, 1],
                           [1, 1, 30.0,-1]],
                          columns=['a','b','c','d'])

        # Test that error is raised if ylabs parameter doesn't contain two unique values
        self.assertRaises(ValueError, munge.match_binary_labels, df, 'a', ylabs=[], rseed=None)
        self.assertRaises(ValueError, munge.match_binary_labels, df, 'a', ylabs=[1,2,3], rseed=None)
        self.assertRaises(ValueError, munge.match_binary_labels, df, 'a', ylabs=[2,2], rseed=None)

        # Test that error is raised if it's not a binary classification problem
        self.assertRaises(ValueError, munge.match_binary_labels, df, 'c')

        # Test that error is raised if the param y-labels are not the same found in the y labels column
        self.assertRaises(ValueError, munge.match_binary_labels, df, 'a', ylabs=[1,2], rseed=None)
        self.assertRaises(ValueError, munge.match_binary_labels, df, 'c', rseed=None)

        # Test that when the two groups are the same size, the same df is returned
        df = pd.DataFrame([[1, 0, 10.0,-1, 1],
                           [0, 1, 20.0, 1, 0]],
                          columns=['a','b','c','d','e'])
        self.assertTrue((munge.match_binary_labels(df, 'e').values == df.values).all())

        # Test sampling from the bigger group to match the smaller group
        df = pd.DataFrame([[1, 0, 10.0,-1, 1],
                           [0, 1, 20.0, 1, 1],
                           [1, 1, 30.0,-1, 1],
                           [0, 0, 24.0,-1, 1],
                           [1, 1, 50.0,-1 ,0],
                           [0, 2, 60.0, 1 ,0]],
                          columns=['a','b','c','d','e'])
        df2 = munge.match_binary_labels(df, 'e')
        self.assertEqual(df2.shape, (4,5))
        self.assertEqual(len(df2[df2.e == 1]), len(df2[df2.e == 0]))

        df = pd.DataFrame([[1, 0, 10.0,-1, 1],
                           [0, 1, 20.0, 1, 1],
                           [1, 1, 30.0,-1, 1],
                           [0, 0, 24.0,-1, 1],
                           [1, 0, 40.0,-1, 0],
                           [1, 1, 50.0,-1 ,0],
                           [0, 2, 60.0, 1 ,0]],
                          columns=['a','b','c','d','e'])
        df2 = munge.match_binary_labels(df, 'e')
        self.assertEqual(df2.shape, (6,5))
        self.assertEqual(len(df2[df2.e == 1]), len(df2[df2.e == 0]))

        df = pd.DataFrame([[1, 0, 10.0,-1, 1],
                           [0, 1, 20.0, 1, 1],
                           [1, 1, 30.0,-1, 1],
                           [1, 1, 23.0, 1, 1],
                           [0, 0, None,-1, 1],
                           [0, 0, None,-1, 1],
                           [0, 0, None,-1, 1],
                           [0, 0, None,-1, 1],
                           [0, 0, None,-1, 1],
                           [1, 0, 40.0,-1, 0],
                           [1, 1, 50.0,-1 ,0],
                           [0, 2, 60.0, 1 ,0]],
                          columns=['a','b','c','d','e'])
        df2 = munge.match_binary_labels(df, 'e')
        self.assertTrue(df2.shape, (6,5))
        self.assertTrue(df2.shape, df2.dropna().shape)
        self.assertEqual(len(df2[df2.e == 1]), len(df2[df2.e == 0]))

        # Test when removing NAs causes the resulting larger group to be equal in size to the smaller
        df = pd.DataFrame([[1, 0, 10.0,-1, 1],
                           [0, 1, 20.0, 1, 1],
                           [1, 1, 30.0,-1, 1],
                           [0, 0, None,-1, 1],
                           [1, 0, 40.0,-1, 0],
                           [1, 1, 50.0,-1 ,0],
                           [0, 2, 60.0, 1 ,0]],
                          columns=['a','b','c','d','e'])
        df2 = pd.DataFrame([[1, 0, 10.0,-1, 1],
                            [0, 1, 20.0, 1, 1],
                            [1, 1, 30.0,-1, 1],
                            [1, 0, 40.0,-1, 0],
                            [1, 1, 50.0,-1 ,0],
                            [0, 2, 60.0, 1 ,0]],
                           columns=['a','b','c','d','e'])
        self.assertTrue((munge.match_binary_labels(df, 'e').values == df2.values).all())

        # Test that the bigger group (group of rows with y-label that are more sizable) will
        # be reduced by first selecting the complete rows, then sampling randomly from the incomplete
        df = pd.DataFrame([[1, 0, 10.0,-1, 1],
                           [0, 1, 20.0, 1, 1],
                           [1, None, 30.0,-1, 1],
                           [0, 0, None,-1, 1],
                           [0, 0, None,-1, 1],
                           [0, None, 22.4,-1, 1],
                           [0, 0, None,-1, 1],
                           [0, 0, None,-1, 1],
                           [0, 0, None,-1, 1],
                           [0, 0, 30.2,None, 1],
                           [0, 0, None,-1, 1],
                           [0, 0, None,-1, 1],
                           [0, 0, 30.2,None, 1],
                           [0, 0, None,-1, 1],
                           [0, 0, None,-1, 1],
                           [0, 0, 30.2,None, 1],
                           [1, 0, 40.0,-1, 0],
                           [0, 1, 51.0,-1 ,0],
                           [1, 0, 52.0,-1 ,0],
                           [0, 1, 53.0,-1 ,0],
                           [0, 2, 60.0, 1 ,0]],
                          columns=['a','b','c','d','e'])
        df2 = munge.match_binary_labels(df, 'e')
        self.assertTrue(df2.shape, (10,5))
        self.assertEqual(len(df2[df2.e == 1]), len(df2[df2.e == 0]))
        self.assertTrue(df2.dropna().shape, (7,5))

    def test_scale_down(self):
        '''
        Test scaling down large numeric values represented as str
        '''
        # Check 0 and negative mvleft parameter value
        self.assertRaises(ValueError, munge.scale_down, '123', mvleft=0)
        self.assertRaises(ValueError, munge.scale_down, '123', mvleft=-1)

        # Check non-numeric strings
        self.assertRaises(ValueError, munge.scale_down, 'abc1')
        self.assertRaises(ValueError, munge.scale_down, '111a22.2')

        # Check None and np.nan
        self.assertEqual(munge.scale_down(None), None)
        self.assertTrue(np.isnan(munge.scale_down(np.nan)))

        # Check that error is raised for non-string values
        self.assertRaises(ValueError, munge.scale_down, 3.4)

        # Check that a string with more than one decimal raises error
        self.assertRaises(ValueError, munge.scale_down, '1.2.3')

        # Check various values
        self.assertEqual(munge.scale_down('1234567'), 1.234567)
        self.assertEqual(munge.scale_down('123456.7'), 0.1234567)
        self.assertEqual(munge.scale_down('.001'), 0.000000001)
        self.assertEqual(munge.scale_down('123'), 0.000123)
        self.assertEqual(munge.scale_down('1'), 0.000001)

        # Change number of decimals
        self.assertEqual(munge.scale_down('1', mvleft=4), 0.0001)
        self.assertEqual(munge.scale_down('1', mvleft=8), 0.00000001)

    def test_scale_down_cols(self):
        '''
        Test applying scale_down on multiple columns in a dataframe
        '''
        df = pd.DataFrame([['123456789', '1234567890123'],
                           ['123456', '1.2']], columns=['a','b'])
        df2 = munge.scale_down_cols(df, cols=['a'], mvleft=3)
        self.assertEqual(df2['a'][0], 123456.789)
        self.assertEqual(df2['a'][1], 123.456)
        self.assertEqual(df2['b'][1], '1.2')

        df3 = munge.scale_down_cols(df, cols=['a','b'], mvleft=9)
        self.assertEqual(df3['a'][0], 0.123456789)
        self.assertEqual(df3['b'][0], 1234.567890123)

    def test_remove_null_big_data(self):
        '''
        Test removing null values
        '''
        sample_file_csv = os.path.join(self.curdir, 'res', 'sample2.csv')
        sample_file_csv_nonull = os.path.join(self.curdir, 'res', 'sample2.nonull.csv')

        # Remove rows containing empty values
        munge.remove_null_big_data(sample_file_csv, sample_file_csv_nonull)

        df = pd.read_csv(sample_file_csv_nonull)
        self.assertEqual(df.shape[0], 5)
        self.assertEqual(list(df['a'].values), [11,1111,111111,1111111,111111111])

    def test_remove_col_big_data(self):
        '''
        Test removing a column
        '''
        sample_file = os.path.join(self.curdir, 'res', 'sample1.txt')
        sample_file_delcol = os.path.join(self.curdir, 'res', 'sample1.delcol.txt')

        def test_removing_idx(idx):
            munge.remove_col_big_data(sample_file, sample_file_delcol, [idx], delimiter='\t')
            data1 = []
            with open(sample_file) as f:
                for line in f:
                    row = line.strip().split()
                    del row[idx]
                    data1.append(row)
            data2 = []
            with open(sample_file_delcol) as f:
                for line in f:
                    data2.append(line.strip().split())
            self.assertEqual(data1, data2)

        for col in range(3):
            test_removing_idx(col)

        # Delete file created
        os.remove(sample_file_delcol)

        # Test removing two columns
        sample_file = os.path.join(self.curdir, 'res', 'sample1.txt')
        sample_file_delcol = os.path.join(self.curdir, 'res', 'sample1.delcol.txt')
        munge.remove_col_big_data(sample_file, sample_file_delcol, [0,2], delimiter='\t')
        data1 = []
        with open(sample_file) as f:
            for line in f:
                row = line.strip().split()
                data1.append([row[1]])
        data2 = []
        with open(sample_file_delcol) as f:
            for line in f:
                data2.append(line.strip().split())
        self.assertEqual(data1, data2)
        os.remove(sample_file_delcol)

    def test_one_hot_encode_features(self):
        '''
        Test one-hot encoding features
        '''
        df = pd.DataFrame([['aa','bb',None],
                           ['aaa','bbb','ccc']],
                          columns=['a','b','c'])
        ohe_df = munge.one_hot_encode_features(df)
        self.assertEqual(ohe_df.shape[0], 2)
        self.assertEqual(ohe_df.shape[1], 5)
        self.assertEqual(ohe_df.ix[0, 'a_aa'], 1)
        self.assertEqual(ohe_df.ix[1, 'b_bb'], 0)
        self.assertEqual(ohe_df.ix[1, 'c_ccc'], 1)

        ohe_df = munge.one_hot_encode_features(df, ['a', 'c'])
        self.assertEqual(ohe_df.shape[0], 2)
        self.assertEqual(ohe_df.shape[1], 4)
        self.assertTrue('b' in ohe_df.columns)
        self.assertEqual(ohe_df.ix[1, 'a_aa'], 0)
        self.assertEqual(ohe_df.ix[1, 'a_aaa'], 1)
        self.assertEqual(ohe_df.ix[0, 'c_ccc'], 0)


if __name__ == '__main__':
    unittest.main()
