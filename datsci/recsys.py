#!/usr/bin/env python
'''
Description     : Module implementing recommender systems
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2014.03.25
Last Modified   : 
Modified By     : 
'''

import numpy as np
import pandas as pd
import scipy as sp
import scipy.spatial

class RecommenderFrame(pd.DataFrame):
    '''
    A recommender system class, extends pandas.DataFrame
    '''

    def __init__(self, *args, **kwargs):
        '''
        Initializer, makes call to pandas DataFrame initializer
        Ensures that the data contains the following columns: user, item, rating
        '''
        super(RecommenderFrame, self).__init__(*args, **kwargs)
        if len({'user', 'item', 'rating'}.intersection(set(self.columns))) != 3:
            raise ValueError, "Column names must contain the following: user, item, rating"


class CollabFilterFrame(RecommenderFrame):
    '''
    Class that implements collaborative filtering, extends Recommender
    Based on the collaborative filtering algorithm presented in
    Programming Collective Intelligence by Toby Segaran
    '''
    
    def similarity(self, u1, u2, method='pearson'):
        '''
        Return similarity(correlation) between two users using various distance functions
        method must be one of {pearson, kendall, spearman, manhattan, euclidean}
        For manhattan and euclidean, must return 1 / (1 + distance)
        since distance 0 indicates high correlation, and vice versa
        '''
        u1_data = self[self['user'] == u1]
        u2_data = self[self['user'] == u2]
        pair = pd.merge(u1_data, u2_data, how='inner', on='item')[['rating_x','rating_y']]
        
        # If there are no overlapping items between two users, return 0
        if pair.shape[0] == 0:
            return 0.0

        if method in {'pearson', 'kendall', 'spearman'}:
            return pair.corr(method=method).ix['rating_x', 'rating_y']
        elif method == 'manhattan':
            d = scipy.spatial.distance.cityblock(pair['rating_x'], pair['rating_y'])
        else:
            # By default, use Euclidean distance
            #d = np.linalg.norm(pair['rating_x'] - pair['rating_y'])
            d = scipy.spatial.distance.euclidean(pair['rating_x'], pair['rating_y'])
        return 1.0 / (1 + d)


    def top_matches(self, user, n=5, method='pearson'):
        '''
        Return n top matching users with given user
        '''
        scores = []
        for u in self['user'].drop_duplicates():
            if u != user:
                s = self.similarity(user, u, method=method)
                scores.append((s, u))
        return sorted(scores, reverse=True)[:n]
