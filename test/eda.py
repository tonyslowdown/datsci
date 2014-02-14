#!/usr/bin/env python
'''
Description     : Unit test for eda.py
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

    def test_find_binary_cols(self):
        '''
        Test finding binary-valued columns in a dataframe
        '''
        df = pd.DataFrame([[1, 11, 1,    1.0,  1.0, 1.000001],
                           [0, 11, None, 0,    1.0, 0],
                           [1, 22, None, None, 1.0, 0]], columns=['a','b','c','d','e','f'])
        self.assertEqual(eda.find_binary_cols(df), ['a','c','d','e'])
        

if __name__ == '__main__':
    unittest.main()
