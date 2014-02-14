#!/usr/bin/env python
'''
Description     : Unit test for datsci.py
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2013.09.23
Last Modified   : 2014.02.12
Modified By     : Jin Kim jjinking(at)gmail(dot)com
'''

import os,sys
# Add project root to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))
import numpy as np
import unittest
from datsci import dataio

class TestIO(unittest.TestCase):
    '''
    Unit tests for the io module
    '''

    def setUp(self):
        '''
        Set up test environment variables
        '''
        self.curdir = os.path.dirname(os.path.abspath(__file__))

    def test_reservoir_sample(self):
        '''
        Test reservoir sampling
        '''
        data = [1,2,3,4,5,6,7,8,9,10]

        # Test subset sampling using random seed
        data_sample = dataio.reservoir_sample(data, 5, rseed=1)
        self.assertEqual(data_sample, [6, 2, 9, 4, 10])

        # Test subset sampling using same random seed as before
        data_sample2 = dataio.reservoir_sample(data, 5, rseed=1)
        self.assertEqual(data_sample, data_sample2)
        
        # Test sampling the entire data
        data_sample = dataio.reservoir_sample(data, len(data))
        self.assertEqual(data_sample, data)

        # Test consecutive subset samplings without a defined seed
        data_sample = dataio.reservoir_sample(data, 7)
        data_sample2 = dataio.reservoir_sample(data, 7)
        self.assertNotEqual(data_sample, data_sample2)
    
    def test_load_subset(self):
        '''
        Test loading a subset of data from filehandle to a dataframe
        '''

        # Setup test parameters
        sample_file = os.path.join(self.curdir, 'res/sample1.csv')
        sample_file_length = 0
        for i,line in enumerate(open(sample_file, 'rU')):
            if i == 0:
                sample_cols = line.strip().split(',')
            sample_file_length += 1
        sample_cols_len = len(sample_cols)
        sample_data_shape = (sample_file_length - 1, sample_cols_len)
        sample_data_nocols_shape = (sample_file_length, sample_cols_len)
        mycols = ['x','y','z']
        autocols = [0, 1, 2]
        
        #===========================================================
        # Reading entire file

        #-----------------------------------------------------------
        # colnames defined

        # k = None
        df = dataio.load_subset(sample_file, colnames=mycols)
        self.assertEqual(df.shape, sample_data_shape)
        self.assertEqual(list(df.columns), mycols)

        # k = length(file)
        df = dataio.load_subset(sample_file, k=sample_file_length, colnames=mycols)
        self.assertEqual(df.shape, sample_data_shape)
        self.assertEqual(list(df.columns), mycols)

        #-----------------------------------------------------------
        # colnames not defined, use first row of file as colnames

        # k = None
        df = dataio.load_subset(open(sample_file, 'rU'))
        self.assertEqual(df.shape, sample_data_shape)
        self.assertEqual(list(df.columns), sample_cols)

        # k > length(file)
        df = dataio.load_subset(sample_file, k=9)
        self.assertEqual(df.shape, sample_data_shape)
        self.assertEqual(list(df.columns), sample_cols)

        #-----------------------------------------------------------
        # header = None, treat entire file as data

        # k = None
        df = dataio.load_subset(open(sample_file, 'rU'), header=None)
        self.assertEqual(df.shape, sample_data_nocols_shape)
        self.assertEqual(list(df.columns), autocols)

        # k = None, colnames defined
        df = dataio.load_subset(sample_file, header=None, colnames=mycols)
        self.assertEqual(df.shape, sample_data_nocols_shape)
        self.assertEqual(list(df.columns), mycols)

        # k = length(file)
        df = dataio.load_subset(sample_file, k=9, header=None)
        self.assertEqual(df.shape, sample_data_nocols_shape)
        self.assertEqual(list(df.columns), autocols)

        # k > length(file), colnames defined
        df = dataio.load_subset(sample_file, k=100, header=None, colnames=mycols)
        self.assertEqual(df.shape, sample_data_nocols_shape)
        self.assertEqual(list(df.columns), mycols)
        
        #===========================================================
        # Reading subset

        # colnames defined
        df = dataio.load_subset(sample_file, k=5, colnames=mycols)
        self.assertEqual(df.shape, (5, sample_cols_len))
        self.assertEqual(list(df.columns), mycols)

        # colnames defined, header = None
        df = dataio.load_subset(sample_file, k=5, colnames=mycols, header=None)
        self.assertEqual(df.shape, (5, sample_cols_len))
        self.assertEqual(list(df.columns), mycols)
        # Test that the first line of file is read in as data
        num_iter = 10 * sample_file_length
        num_1st_line = 0
        for i in range(num_iter):
            df = dataio.load_subset(sample_file, k=sample_file_length/2, colnames=mycols, header=None)
            if sample_cols[0] in df[mycols[0]].values:
                num_1st_line += 1
        self.assertTrue(0 < num_1st_line < num_iter)

        # colnames = None
        df = dataio.load_subset(sample_file, k=5)
        self.assertEqual(df.shape, (5, sample_cols_len))
        self.assertEqual(list(df.columns), sample_cols)

        # colnames = None, header = None
        df = dataio.load_subset(sample_file, k=5, header=None)
        self.assertEqual(df.shape, (5, sample_cols_len))
        self.assertEqual(list(df.columns), autocols)
        # Test that the first line of file is read in as data
        num_1st_line = 0
        for i in range(num_iter):
            df = dataio.load_subset(sample_file, k=sample_file_length/2, header=None)
            if sample_cols[0] in df[autocols[0]].values:
                num_1st_line += 1
        self.assertTrue(0 < num_1st_line < num_iter)


        #===========================================================
        # Delimiters and rseed

        sample_file_txt = os.path.join(self.curdir, 'res/sample1.txt')

        # sep = '\t', k = None
        df = dataio.load_subset(sample_file_txt, sep='\t')
        self.assertEqual(df.shape, sample_data_shape)
        self.assertEqual(list(df.columns), sample_cols)

        # sep = '\t', k = 5
        df = dataio.load_subset(open(sample_file_txt, 'rU'), sep='\t', k=5, header=None)
        self.assertEqual(df.shape, (5, sample_cols_len))
        self.assertEqual(list(df.columns), autocols)

        # rseed = 1, k = 5
        df = dataio.load_subset(open(sample_file_txt, 'rU'), sep='\t', k=5, header=None, rseed=1)
        self.assertEqual(df.shape, (5, sample_cols_len))
        self.assertEqual(list(df.columns), autocols)
        df2 = dataio.load_subset(open(sample_file_txt, 'rU'), sep='\t', k=5, header=None, rseed=1)
        a1 = np.squeeze(df.values.reshape(np.multiply(*df.shape),1))
        a2 = np.squeeze(df2.values.reshape(np.multiply(*df.shape),1))
        self.assertTrue((a1 == a2).all())
        
        # rseed = None, k = 5
        df = dataio.load_subset(open(sample_file_txt, 'rU'), sep='\t', k=5, header=None)
        df2 = dataio.load_subset(open(sample_file_txt, 'rU'), sep='\t', k=5, header=None)
        a1 = np.squeeze(df.values.reshape(np.multiply(*df.shape),1))
        a2 = np.squeeze(df2.values.reshape(np.multiply(*df.shape),1))
        self.assertFalse((a1 == a2).all())


if __name__ == '__main__':
    unittest.main()
