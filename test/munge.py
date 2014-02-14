#!/usr/bin/env python
'''
Description     : Unit test for munge.py
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2014.02.13
Last Modified   : 2014.02.14
Modified By     : Jin Kim jjinking(at)gmail(dot)com
'''

import os,sys
# Add project root to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))
import numpy as np
import pandas as pd
import unittest
from datsci import munge

class TestMunge(unittest.TestCase):
    '''
    Unit tests for the munge module
    '''

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


if __name__ == '__main__':
    unittest.main()
