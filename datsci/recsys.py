"""Recommender systems

"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2014.03.25
# Last Modified   : 2014.03.27
#
# License         : MIT

import numpy as np
import pandas as pd
import scipy as sp
import scipy.spatial
from collections import defaultdict


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

    def create_matrix(self):
        '''
        Create matrix where the rows are the users and the columns are the items,
        and each element is a rating
        '''
        m = pd.DataFrame(self.values, columns=self.columns)
        m.set_index(['user','item'], inplace=True)
        m = m.unstack('item')
        m.columns = m.columns.get_level_values(1)
        return m


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

    def get_user_matches(self, user, n=None, method='pearson'):
        '''
        Return n top matching users with given user
        If n is not set, return all users sorted by similarity scores from highest to lowest
        '''
        scores = []
        for u in self['user'].drop_duplicates():
            if u != user:
                s = self.similarity(user, u, method=method)
                scores.append((s, u))
        return sorted(scores, reverse=True)[:n]

    def get_recommendations(self, user, n=None, method='pearson'):
        '''
        Return n top recommended items for given user
        If n is not set, return all items sorted by recommendation scores from highest to lowest
        '''
        # Create dataframe of users and scores
        rec_table = pd.DataFrame([(u,s) for s,u in self.get_user_matches(user, method=method)],
                                 columns=['user', 'score'])
        rec_table.set_index('user', inplace=True)

        # Remove users with similarity score <= 0
        rec_table = rec_table[rec_table['score'] > 0]

        # Merge with user ratings table
        rec_table = rec_table.join(self.create_matrix())

        # Create weighted scores of unseen items
        items_seen = set(self[self['user'] == user]['item'].dropna().drop_duplicates())
        items = list(set(self['item'].drop_duplicates()) - items_seen)
        sim_sum = defaultdict(float)
        for item in items:
            rec_table[item] = rec_table.apply(lambda row: row[item] * row['score'], axis=1)
            sim_sum[item] += rec_table[rec_table[item].notnull()]['score'].sum()

        # Create dataframe of rec_scores
        rec_scores = pd.DataFrame(rec_table[items].sum(), columns=['rec_score'])

        # Correct each recommendation score by the sum of the similarity scores of all users who rated the item
        rec_scores['simsum'] = rec_scores.apply(lambda row: sim_sum[row.name], axis=1)
        rec_scores['rec_score_corrected'] = rec_scores.apply(lambda row: row['rec_score'] / row['simsum'], axis=1)
        item_scores = rec_scores['rec_score_corrected'].to_dict().items()
        return sorted([(None if np.isnan(rec_score) else rec_score, item) for item,rec_score in item_scores],
                      reverse=True)[:n]
