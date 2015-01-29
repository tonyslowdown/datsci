#!/usr/bin/env python
'''
Description     : Unit test for datsci.py
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2013.09.23
Last Modified   : 2014.09.20
Modified By     : Jin Kim jjinking(at)gmail(dot)com
'''

import os,sys
# Add project root to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))
import numpy as np
import pandas as pd
import unittest
from datsci import dataio, eda

class TestIO(unittest.TestCase):
    '''
    Unit tests for the io module
    '''

    def setUp(self):
        '''
        Set up test environment variables
        '''
        self.curdir = os.path.dirname(os.path.abspath(__file__))

    def test_fopen(self):
        '''
        Test getting file handle based on file type
        '''
        import gzip
        import tarfile
        import zipfile

        ######################################################
        # Reading from file

        # txt file
        sample_file = os.path.join(self.curdir, 'res', 'sample1.txt')
        f = dataio.fopen(sample_file)
        self.assertTrue(isinstance(f, file))
        self.assertEqual(f.next().strip().split('\t')[0], 'a')
        self.assertEqual(f.next().strip().split('\t')[1], '2')
        self.assertEqual(f.next().strip().split('\t')[2], '33')

        # csv file
        sample_file = os.path.join(self.curdir, 'res', 'sample1.csv')
        f = dataio.fopen(sample_file)
        self.assertTrue(isinstance(f, file))
        self.assertEqual(f.next().strip().split(',')[0], 'a')
        self.assertEqual(f.next().strip().split(',')[1], '2')
        self.assertEqual(f.next().strip().split(',')[2], '33')

        # tsv file
        sample_file = os.path.join(self.curdir, 'res', 'sample1.tsv')
        f = dataio.fopen(sample_file)
        self.assertTrue(isinstance(f, file))
        self.assertEqual(f.next().strip().split('\t')[0], 'a')
        self.assertEqual(f.next().strip().split('\t')[1], '2')
        self.assertEqual(f.next().strip().split('\t')[2], '33')

        # zip file
        sample_file = os.path.join(self.curdir, 'res', 'sample1.csv.zip')
        f = dataio.fopen(sample_file)
        self.assertTrue(isinstance(f, zipfile.ZipExtFile))
        self.assertEqual(f.next().strip().split(',')[0], 'a')
        self.assertEqual(f.next().strip().split(',')[1], '2')
        self.assertEqual(f.next().strip().split(',')[2], '33')
    
        # gz file
        sample_file = os.path.join(self.curdir, 'res', 'sample1.csv.gz')
        f = dataio.fopen(sample_file)
        self.assertTrue(isinstance(f, gzip.GzipFile))
        self.assertEqual(f.next().strip().split(',')[0], 'a')
        self.assertEqual(f.next().strip().split(',')[1], '2')
        self.assertEqual(f.next().strip().split(',')[2], '33')

        # tar.gz
        sample_file = os.path.join(self.curdir, 'res', 'sample1.csv.tar.gz')
        f = dataio.fopen(sample_file)
        self.assertTrue(isinstance(f, tarfile.ExFileObject))
        self.assertEqual(f.readline().strip().split(',')[0], 'a')
        self.assertEqual(f.readline().strip().split(',')[1], '2')
        self.assertEqual(f.readline().strip().split(',')[2], '33')

        # tar.bz2
        sample_file = os.path.join(self.curdir, 'res', 'sample1.csv.tar.bz2')
        f = dataio.fopen(sample_file)
        self.assertTrue(isinstance(f, tarfile.ExFileObject))
        self.assertEqual(f.readline().strip().split(',')[0], 'a')
        self.assertEqual(f.readline().strip().split(',')[1], '2')
        self.assertEqual(f.readline().strip().split(',')[2], '33')

        ######################################################
        # Writing to file

        # tsv file
        # Read from file
        sample_file = os.path.join(self.curdir, 'res', 'sample1.tsv')
        data = dataio.fopen(sample_file).read()
        # Write to second duplicate file
        sample_outfile = os.path.join(self.curdir, 'res', 'sample1.dup.tsv')
        with dataio.fopen(sample_outfile, 'w') as fout:
            self.assertTrue(isinstance(fout, file))
            fout.write(data)
        # Read from second duplicate file
        data2 = open(sample_outfile, 'r').read()
        self.assertEqual(data, data2)
        os.remove(sample_outfile)

        # gz file
        sample_file = os.path.join(self.curdir, 'res', 'sample1.csv.gz')
        data = dataio.fopen(sample_file).read()
        # Write to second duplicate file
        sample_outfile = os.path.join(self.curdir, 'res', 'sample1.csv.dup.gz')
        with dataio.fopen(sample_outfile, 'w') as fout:
            self.assertTrue(isinstance(fout, gzip.GzipFile))
            fout.write(data)
        # Read from second duplicate file
        data2 = dataio.fopen(sample_outfile, 'r').read()
        self.assertEqual(data, data2)
        os.remove(sample_outfile)

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
        sample_file = os.path.join(self.curdir, 'res', 'sample1.csv')
        sample_file_length = 0
        for i,line in enumerate(dataio.fopen(sample_file, 'r')):
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

        sample_file_txt = os.path.join(self.curdir, 'res', 'sample1.txt')

        # sep = '\t', k = None
        df = dataio.load_subset(sample_file_txt, sep='\t')
        self.assertEqual(df.shape, sample_data_shape)
        self.assertEqual(list(df.columns), sample_cols)

        # sep = '\t', k = 5
        df = dataio.load_subset(sample_file_txt, sep='\t', k=5, header=None)
        self.assertEqual(df.shape, (5, sample_cols_len))
        self.assertEqual(list(df.columns), autocols)

        # rseed = 1, k = 5
        df = dataio.load_subset(sample_file_txt, sep='\t', k=5, header=None, rseed=1)
        self.assertEqual(df.shape, (5, sample_cols_len))
        self.assertEqual(list(df.columns), autocols)
        df2 = dataio.load_subset(sample_file_txt, sep='\t', k=5, header=None, rseed=1)
        a1 = np.squeeze(df.values.reshape(np.multiply(*df.shape),1))
        a2 = np.squeeze(df2.values.reshape(np.multiply(*df.shape),1))
        self.assertTrue((a1 == a2).all())
        
        # rseed = None, k = 5
        df = dataio.load_subset(sample_file_txt, sep='\t', k=5, header=None)
        df2 = dataio.load_subset(sample_file_txt, sep='\t', k=5, header=None)
        a1 = np.squeeze(df.values.reshape(np.multiply(*df.shape),1))
        a2 = np.squeeze(df2.values.reshape(np.multiply(*df.shape),1))
        self.assertFalse((a1 == a2).all())

    def test_head(self):
        '''
        Test loading first few lines of file
        '''
        sample_file_txt = os.path.join(self.curdir, 'res', 'sample1.txt')
        colnames = dataio.head(sample_file_txt, k=0, sep='\t')
        self.assertTrue(isinstance(colnames, pd.Series))

        sample_file = os.path.join(self.curdir, 'res', 'sample1.csv.gz')
        df = dataio.head(sample_file, k=1)
        self.assertEqual(df.shape[0], 1)
        self.assertEqual(df.shape[1], 3)
        df = dataio.head(sample_file, k=7)
        self.assertEqual(df.shape[0], 7)
        df = dataio.head(sample_file, k=8, sep=',')
        self.assertEqual(df.shape[0], 8)
        df = dataio.head(sample_file, k=9)
        self.assertEqual(df.shape[0], 8)
        df = dataio.head(sample_file, k=100, sep=',')
        self.assertEqual(df.shape[0], 8)

        # Must change to numeric, since data read from file is string
        for col in df.columns:
            df[col] = df[col].astype(int)
        df2 = pd.read_csv(sample_file_txt, sep='\t')
        self.assertTrue(eda.df_equal(df, df2))




if __name__ == '__main__':
    unittest.main()
