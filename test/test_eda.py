"""Unit test for eda.py
"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2014.02.13
# Last Modified   : 2016.04.17
#
# License         : MIT

import os
import numpy as np
import pandas as pd
import scipy.stats
import unittest
from datsci import eda
from sklearn.ensemble import RandomForestClassifier

CURDIR = os.path.dirname(os.path.abspath(__file__))


class TestEda(unittest.TestCase):

    def test_df_isclose(self):

        # Test integers
        df1 = pd.DataFrame([[1, 2],
                            [3, 4]])
        df2 = pd.DataFrame([[1, 2],
                            [3, 4]])
        self.assertTrue(eda.df_isclose(df1, df2, tol=0))

        df1 = pd.DataFrame([[1, 2],
                            [3, 5]])
        df2 = pd.DataFrame([[1, 2],
                            [3, 4]])
        self.assertFalse(eda.df_isclose(df1, df2))

        # Test rounding
        df1 = pd.DataFrame([[1.1234, 2.1234],
                            [3.1234, 4.1234]])
        df2 = pd.DataFrame([[1.1234, 2.1234],
                            [3.1234, 4.1234]])
        self.assertTrue(eda.df_isclose(df1, df2))

        df1 = pd.DataFrame([[1.1234, 2.1234],
                            [3.1234, 5.1234]])
        df2 = pd.DataFrame([[1.1234, 2.1234],
                            [3.1234, 4.1232]])
        self.assertFalse(eda.df_isclose(df1, df2))

        df1 = pd.DataFrame([[1.1234, 2.1234],
                            [3.1234, 4.1234]])
        df2 = pd.DataFrame([[1.1234, 2.1234],
                            [3.1234, 4.1232]])
        self.assertTrue(eda.df_isclose(df1, df2, decimals=3))

        df1 = pd.DataFrame([[np.nan, 2.1234],
                            [3.1234, 5.1234123]])
        df2 = pd.DataFrame([[np.nan, 2.1234],
                            [3.1234, 5.123412]])
        self.assertTrue(eda.df_isclose(df1, df2, decimals=6))

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

    def test_summarize_training_data(self):
        '''
        Test file summarization
        '''
        sample_file_csv = os.path.join(CURDIR, 'res', 'sample1.csv')
        df = pd.read_csv(sample_file_csv)
        (summary,
         n_rows,
         label_counts) = eda.summarize_training_data(df,
                                                     y_name='c',
                                                     summary_pkl=None)
        self.assertEqual(n_rows, 8)
        self.assertEqual(len(label_counts), 8)
        self.assertEqual(set(label_counts.values()), {1})
        self.assertEqual(summary.shape[0], 3)
        self.assertEqual(summary[summary['attribute']=='a']['min'].values[0], 1)
        self.assertEqual(summary[summary['attribute']=='a']['max'].values[0], 11111111)
        self.assertEqual(summary[summary['attribute']=='a']['n_null'].values[0], 0)
        self.assertEqual(summary[summary['attribute']=='a']['perc_null'].values[0], 0)
        self.assertEqual(summary[summary['attribute']=='b']['min'].values[0], 2)
        self.assertEqual(summary[summary['attribute']=='b']['max'].values[0], 22222222)
        self.assertEqual(summary[summary['attribute']=='b']['n_null'].values[0], 0)
        self.assertEqual(summary[summary['attribute']=='b']['perc_null'].values[0], 0)
        self.assertEqual(summary[summary['attribute']=='c']['min'].values[0], 3)
        self.assertEqual(summary[summary['attribute']=='c']['max'].values[0], 33333333)
        self.assertEqual(summary[summary['attribute']=='c']['n_null'].values[0], 0)
        self.assertEqual(summary[summary['attribute']=='c']['perc_null'].values[0], 0)

        sample_file_csv = os.path.join(CURDIR, 'res', 'sample2.csv')
        df = pd.read_csv(sample_file_csv)
        (summary,
         n_rows,
         label_counts) = eda.summarize_training_data(df,
                                                     y_name='c',
                                                     summary_pkl=None)
        self.assertEqual(n_rows, 10)
        self.assertEqual(len(label_counts), 4)
        self.assertEqual(set(label_counts.values()), {1,2,3,4})
        self.assertEqual(label_counts[sorted(label_counts.keys())[0]], 3)
        self.assertEqual(label_counts[3], 1)
        self.assertEqual(label_counts[33], 2)
        self.assertEqual(label_counts[333], 4)
        self.assertEqual(summary.shape[0], 3)
        summary_a = summary[summary['attribute']=='a']
        self.assertEqual(summary_a['min'].values[0], 11)
        self.assertEqual(summary_a['max'].values[0], 1111111111)
        self.assertEqual(summary_a['n_null'].values[0], 1)
        self.assertEqual(summary_a['perc_null'].values[0], .10)
        self.assertEqual(summary_a['n_uniq'].values[0], 10)
        summary_b = summary[summary['attribute']=='b']
        self.assertEqual(summary_b['min'].values[0], 2)
        self.assertEqual(summary_b['max'].values[0], 222222222)
        self.assertEqual(summary_b['n_null'].values[0], 2)
        self.assertEqual(summary_b['perc_null'].values[0], .2)
        self.assertEqual(summary_b['n_uniq'].values[0], 9)
        summary_c = summary[summary['attribute']=='c']
        self.assertEqual(summary_c['min'].values[0], 3)
        self.assertEqual(summary_c['max'].values[0], 333)
        self.assertEqual(summary_c['n_null'].values[0], 3)
        self.assertEqual(summary_c['perc_null'].values[0], .3)
        self.assertEqual(summary_c['n_uniq'].values[0], 4)

        sample_file_csv = os.path.join(CURDIR, 'res', 'sample3.csv')
        summary_pkl_file = os.path.join(CURDIR, 'res', 'foo.pkl')
        df = pd.read_csv(sample_file_csv)
        (summary,
         n_rows,
         label_counts) = eda.summarize_training_data(df,
                                                     y_name='z',
                                                     summary_pkl=summary_pkl_file)
        self.assertEqual(n_rows, 10)
        self.assertEqual(len(label_counts), 4)
        self.assertEqual(set(label_counts.values()), {1,2,3,4})
        self.assertEqual(label_counts[sorted(label_counts.keys())[0]], 3)
        self.assertEqual(label_counts['c'], 1)
        self.assertEqual(label_counts['cc'], 2)
        self.assertEqual(label_counts['ccc'], 4)
        self.assertEqual(summary.shape[0], 3)
        summary_x = summary[summary['attribute']=='x']
        self.assertTrue(pd.isnull(summary_x['min'].values[0]))
        self.assertTrue(pd.isnull(summary_x['max'].values[0]))
        self.assertEqual(summary_x['n_null'].values[0], 1)
        self.assertEqual(summary_x['perc_null'].values[0], .10)
        self.assertEqual(summary_x['n_uniq'].values[0], 10)
        summary_y = summary[summary['attribute']=='y']
        self.assertTrue(pd.isnull(summary_y['min'].values[0]))
        self.assertTrue(pd.isnull(summary_y['max'].values[0]))
        self.assertEqual(summary_y['n_null'].values[0], 2)
        self.assertEqual(summary_y['perc_null'].values[0], .2)
        self.assertEqual(summary_y['n_uniq'].values[0], 9)
        summary_z = summary[summary['attribute']=='z']
        self.assertTrue(pd.isnull(summary_z['min'].values[0]))
        self.assertTrue(pd.isnull(summary_z['max'].values[0]))
        self.assertEqual(summary_z['n_null'].values[0], 3)
        self.assertEqual(summary_z['perc_null'].values[0], .3)
        self.assertEqual(summary_z['n_uniq'].values[0], 4)

        # Check that summary pkl file exists
        self.assertTrue(os.path.exists(summary_pkl_file))

        # Check saved values can be loaded and is correct
        summary2, n_rows2, label_counts2 = eda.load_summary_data(summary_pkl_file)
        self.assertTrue(eda.df_isclose(summary, summary2))
        self.assertEqual(n_rows, n_rows2)
        self.assertEqual(str(list(label_counts.items())),
                         str(list(label_counts2.items())))

        # Delete file
        os.remove(summary_pkl_file)
        self.assertFalse(os.path.exists(summary_pkl_file))

        # Run again with summary_pkl option set to None
        (summary,
         n_rows,
         label_counts) = eda.summarize_training_data(df,
                                                     y_name='z',
                                                     summary_pkl=None)
        self.assertFalse(os.path.exists(summary_pkl_file))

    def test_summarize_big_training_data(self):
        '''
        Test large data file summarization
        '''
        for fname in ['sample1.csv', 'sample1.csv.zip', 'sample1.csv.gz',
                      'sample1.csv.tar.gz', 'sample1.csv.tar.bz2']:
            sample_file_csv = os.path.join(CURDIR, 'res', fname)
            (summary,
             n_rows,
             label_counts) = eda.summarize_big_training_data(sample_file_csv,
                                                             y_name='c',
                                                             summary_pkl=None)

            self.assertEqual(n_rows, 8)
            self.assertEqual(len(label_counts), 8)
            self.assertEqual(set(label_counts.values()), {1})
            self.assertEqual(summary.shape[0], 3)
            summary_a = summary[summary['attribute']=='a']
            self.assertEqual(summary_a['min'].values[0], 1)
            self.assertEqual(summary_a['max'].values[0], 11111111)
            self.assertEqual(summary_a['n_null'].values[0], 0)
            self.assertEqual(summary_a['perc_null'].values[0], 0)
            summary_b = summary[summary['attribute']=='b']
            self.assertEqual(summary_b['min'].values[0], 2)
            self.assertEqual(summary_b['max'].values[0], 22222222)
            self.assertEqual(summary_b['n_null'].values[0], 0)
            self.assertEqual(summary_b['perc_null'].values[0], 0)
            summary_c = summary[summary['attribute']=='c']
            self.assertEqual(summary_c['min'].values[0], 3)
            self.assertEqual(summary_c['max'].values[0], 33333333)
            self.assertEqual(summary_c['n_null'].values[0], 0)
            self.assertEqual(summary_c['perc_null'].values[0], 0)


        sample_file_csv = os.path.join(CURDIR, 'res', 'sample2.csv')
        summary_pkl_file = os.path.join(CURDIR, 'res', 'foo.pkl')
        (summary,
         n_rows,
         label_counts) = eda.summarize_big_training_data(sample_file_csv,
                                                         y_name='c',
                                                         summary_pkl=summary_pkl_file)

        self.assertEqual(n_rows, 10)
        self.assertEqual(len(label_counts), 4)
        self.assertEqual(set(label_counts.values()), {1,2,3,4})
        self.assertEqual(label_counts[''], 3)
        self.assertEqual(label_counts['3'], 1)
        self.assertEqual(label_counts['33'], 2)
        self.assertEqual(label_counts['333'], 4)
        self.assertEqual(summary.shape[0], 3)
        summary_a = summary[summary['attribute']=='a']
        self.assertEqual(summary_a['min'].values[0], 11)
        self.assertEqual(summary_a['max'].values[0], 1111111111)
        self.assertEqual(summary_a['n_null'].values[0], 1)
        self.assertEqual(summary_a['perc_null'].values[0], .10)
        self.assertEqual(summary_a['n_uniq'].values[0], 10)
        summary_b = summary[summary['attribute']=='b']
        self.assertEqual(summary_b['min'].values[0], 2)
        self.assertEqual(summary_b['max'].values[0], 222222222)
        self.assertEqual(summary_b['n_null'].values[0], 2)
        self.assertEqual(summary_b['perc_null'].values[0], .2)
        self.assertEqual(summary_b['n_uniq'].values[0], 9)
        summary_c = summary[summary['attribute']=='c']
        self.assertEqual(summary_c['min'].values[0], 3)
        self.assertEqual(summary_c['max'].values[0], 333)
        self.assertEqual(summary_c['n_null'].values[0], 3)
        self.assertEqual(summary_c['perc_null'].values[0], .3)
        self.assertEqual(summary_c['n_uniq'].values[0], 4)

        # Check that summary pkl file exists
        self.assertTrue(os.path.exists(summary_pkl_file))

        # Check saved values can be loaded and is correct
        summary2, n_rows2, label_counts2 = eda.load_summary_data(summary_pkl_file)
        self.assertTrue(eda.df_isclose(summary, summary2))
        self.assertEqual(n_rows, n_rows2)
        self.assertEqual(set(label_counts.items()),
                         set(label_counts2.items()))

        # Delete file
        os.remove(summary_pkl_file)
        self.assertFalse(os.path.exists(summary_pkl_file))

        # Run again with summary_pkl option set to None
        (summary,
         n_rows,
         label_counts) = eda.summarize_big_training_data(sample_file_csv,
                                                         y_name='c',
                                                         summary_pkl=None)
        self.assertFalse(os.path.exists(summary_pkl_file))

    def test_count_big_file_value_counts(self):
        '''
        Test counting column values in file
        '''
        sample_file_csv = os.path.join(CURDIR, 'res', 'sample4.csv')
        value_counts = eda.count_big_file_value_counts(sample_file_csv, 'a')
        self.assertEqual(value_counts['1'], 4)
        self.assertEqual(value_counts['0'], 2)
        value_counts = eda.count_big_file_value_counts(sample_file_csv, 'b')
        self.assertEqual(value_counts['x'], 3)
        self.assertEqual(value_counts['y'], 3)
        value_counts = eda.count_big_file_value_counts(sample_file_csv, 'c')
        self.assertEqual(value_counts['0'], 6)


if __name__ == '__main__':
    unittest.main()
