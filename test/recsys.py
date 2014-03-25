#!/usr/bin/env python
'''
Description     : Unit test for recsys.py
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2014.03.25
Last Modified   : 
Modified By     : 
'''

import os,sys
# Add project root to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))
import pandas as pd
import unittest
from datsci import recsys

class TestRecommenderFrame(unittest.TestCase):
    '''
    Unit tests for the RecommenderFrame class
    '''

    def test_init(self):
        '''
        Test initializer method
        '''
        data = [['1','2','3'],['1','2','3']]
        self.failUnlessRaises(ValueError, recsys.RecommenderFrame, data)
        
        recframe = recsys.RecommenderFrame(data, columns=['user','item','rating'])
        self.assertTrue(len({'user', 'item', 'rating'}.intersection(set(recframe.columns))), 3)


class TestCollabFilterFrame(unittest.TestCase):
    '''
    Unit tests for the CollabFilterFrame class
    '''

    def setUp(self):
        '''
        Set up test data
        '''
        # Data from Programming Collective Intelligence by Toby Segaran
        data = {'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
                              'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
                              'The Night Listener': 3.0},
                'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
                                 'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
                                 'You, Me and Dupree': 3.5},
                'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
                                     'Superman Returns': 3.5, 'The Night Listener': 4.0},
                'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
                                 'The Night Listener': 4.5, 'Superman Returns': 4.0,
                                 'You, Me and Dupree': 2.5},
                'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                                  'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
                                 'You, Me and Dupree': 2.0},
                'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                                  'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
                'Toby': {'Snakes on a Plane': 4.5,'You, Me and Dupree': 1.0,'Superman Returns': 4.0},
                'Bob': {'Batman': 5.0},
                'Joe': {'Superman': 3.0}}

        # Convert data to dataframe containing user, item, rating columns
        #        user            item                  rating
        # 0      Jack Matthews   Lady in the Water     3.0
        # 1      Jack Matthews   Snakes on a Plane     4.0
        # 2      Jack Matthews  You, Me and Dupree     3.5
        self.recframe = recsys.CollabFilterFrame([(user, item, rating) for user,item_rating in data.iteritems()
                                                  for item,rating in item_rating.iteritems()],
                                                 columns=['user', 'item', 'rating'])

    def test_similarity(self):
        '''
        Test similarity method
        '''
        dist_pearson = self.recframe.similarity('Lisa Rose', 'Gene Seymour', method='pearson')
        self.assertEqual(round(dist_pearson, 12), 0.396059017191)

        dist_euclidean = self.recframe.similarity('Lisa Rose', 'Gene Seymour', method='euclidean')
        self.assertEqual(round(dist_euclidean, 12), 0.294298055086)

        dist_manhattan = self.recframe.similarity('Lisa Rose', 'Gene Seymour', method='manhattan')
        self.assertEqual(round(dist_manhattan, 12), 0.181818181818)

        #TODO Test if no overlaps, return 0
        dist_nooverlaps = self.recframe.similarity('Bob', 'Joe')
        self.assertEqual(dist_nooverlaps, 0)

    def test_top_matches(self):
        '''
        Test top_matches method
        '''
        tm = self.recframe.top_matches('Toby', n=3)
        self.assertEqual([(round(item[0], 11), item[1]) for item in tm],
                         [(0.99124070716, 'Lisa Rose'),
                          (0.92447345164, 'Mick LaSalle'),
                          (0.89340514744, 'Claudia Puig')])

        self.assertEqual(len(self.recframe.top_matches('Toby', n=5)), 5)


if __name__ == '__main__':
    unittest.main()
