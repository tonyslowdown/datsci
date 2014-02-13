#!/usr/bin/env python
'''
Description     : Unit test for eda.py
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
from datsci import eda

class TestEda(unittest.TestCase):
    '''
    Unit tests for the eda module
    '''

    def test_scale_down(self):
        '''
        Test scaling down large numeric values represented as str
        '''
        df = pd.DataFrame([[1, 2, 33, 4],
                           [None, None, np.nan, 4],
                           [None, np.nan, np.nan, 44],
                           [None, 2, 33, 44],
                           [None, np.nan, 33, 44],
                           [None, None, 33, 44]], columns=['a','b','c','d'])
        self.assertEqual(eda.find_uninfo_cols(df), ['a','b','c'])


if __name__ == '__main__':
    unittest.main()
