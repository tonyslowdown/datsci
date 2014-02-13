#!/usr/bin/env python
'''
Description     : Unit test for munge.py
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2014.02.13
Last Modified   : 
Modified By     : 
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
