"""Unit tests for munge.py
"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2014.02.13
# Last Modified   : 2016.04.15
#
# License         : MIT

import os
import pandas as pd
import unittest

from datsci import munge


class TestMunge(unittest.TestCase):

    def setUp(self):
        self.curdir = os.path.dirname(os.path.abspath(__file__))

    def test_remove_null_big_data(self):
        sample_file_csv = os.path.join(self.curdir, 'res', 'sample2.csv')
        sample_file_csv_nonull = os.path.join(
            self.curdir, 'res', 'sample2.nonull.csv')

        # Remove rows containing empty values
        munge.remove_null_big_data(sample_file_csv, sample_file_csv_nonull)

        df = pd.read_csv(sample_file_csv_nonull)
        self.assertEqual(df.shape[0], 5)
        self.assertEqual(list(df['a'].values),
                         [11, 1111, 111111, 1111111, 111111111])

    def test_remove_col_big_data(self):
        sample_file = os.path.join(self.curdir, 'res', 'sample1.txt')
        sample_file_delcol = os.path.join(
            self.curdir, 'res', 'sample1.delcol.txt')

        def test_removing_idx(idx):
            munge.remove_col_big_data(
                sample_file, sample_file_delcol, [idx], delimiter='\t')
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

    def test_one_hot_encode(self):
        df = pd.DataFrame([['aa', 'bb', None],
                           ['aaa', 'bbb', 'ccc']],
                          columns=['a', 'b', 'c'])
        ohe_df = munge.one_hot_encode(df)
        self.assertEqual(ohe_df.shape[0], 2)
        self.assertEqual(ohe_df.shape[1], 5)
        self.assertEqual(ohe_df.ix[0, 'onehot_a_aa'], 1)
        self.assertEqual(ohe_df.ix[1, 'onehot_b_bb'], 0)
        self.assertEqual(ohe_df.ix[1, 'onehot_c_ccc'], 1)

        ohe_df = munge.one_hot_encode(df, ['a', 'c'], prefix='oh_')
        self.assertEqual(ohe_df.shape[0], 2)
        self.assertEqual(ohe_df.shape[1], 4)
        self.assertTrue('b' in ohe_df.columns)
        self.assertEqual(ohe_df.ix[1, 'oh_a_aa'], 0)
        self.assertEqual(ohe_df.ix[1, 'oh_a_aaa'], 1)
        self.assertEqual(ohe_df.ix[0, 'oh_c_ccc'], 0)

    def test_balanced_subsets_basic(self):
        """Test balanced_subsets with basic params and no shuffling
        """
        df = pd.DataFrame([
            [1, 11, 1],
            [2, 22, 1],
            [3, 33, 0],
            [4, 44, 0],
            [5, 55, 0],
            [6, 66, 0]], columns=['a', 'b', 'y'])
        ss = munge.balanced_subsets(df[['a', 'b']], df.y, shuffle=False)

        self.assertEqual(len(ss), 2)

        X1, y1 = ss[0]
        X2, y2 = ss[1]

        # Check equal number of labels per subset
        self.assertTrue((y1 == 1).sum() == (y1 == 0).sum() == 2)
        self.assertTrue((y2 == 1).sum() == (y2 == 0).sum() == 2)

        # Check unshuffled results
        self.assertEqual(set(X1.a.unique()), {1, 2, 3, 4})
        self.assertEqual(set(X2.a.unique()), {1, 2, 5, 6})

        # Test with subsample
        ss = munge.balanced_subsets(
            df[['a', 'b']], df.y, subsample=0.5, shuffle=False)

        self.assertEqual(len(ss), 4)

        X1, y1 = ss[0]
        X3, y3 = ss[2]
        self.assertTrue(X1.shape[0] == X3.shape[0] == 2)

    def test_balanced_subsets_advanced(self):
        """Test balanced_subsets with advanced params
        """
        df = pd.DataFrame([
            [1, 11, 'true'],
            [2, 22, 'true'],
            [3, 33, 'false'],
            [4, 44, 'false'],
            [5, 55, 'false'],
            [6, 66, 'false'],
            [7, 77, 'false']], columns=['a', 'b', 'y'])

        ss = munge.balanced_subsets(
            df[['a', 'b']], df.y, labels=["true", "false"])

        self.assertEqual(len(ss), 2)

        X1, y1 = ss[0]
        X2, y2 = ss[1]
        self.assertEqual(X1.shape[0], 4)
        self.assertEqual(X2.shape[0], 5)

        self.assertTrue((y1 == 'true').sum() == (y1 == 'false').sum() == 2)
        self.assertEqual((y2 == 'true').sum(), 2)
        self.assertEqual((y2 == 'false').sum(), 3)


if __name__ == '__main__':
    unittest.main()
