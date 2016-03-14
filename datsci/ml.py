'''
Description     : Module to run machine learning algorithms
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2016.03.14
Last Modified   : 2016.03.14
Modified By     : Jin Kim jjinking(at)gmail(dot)com
'''

import pandas as pd
import time
from sklearn.metrics import accuracy_score


def train_predict(descriptions_clfs,
                  X_train, y_train,
                  X_test, y_test,
                  scoring=accuracy_score):
    '''
    Function for running preliminary analyses

    descriptions_clfs is a list of 2-tuples, each containing a
    description text and a classifier object
    i.e. [('Classifier1 info', clf1), ('Classifier2 info', clf2), ...]
    '''
    results = []
    for description, clf in descriptions_clfs:

        result = {'description': description}

        # Train
        start = time.time()
        clf.fit(X_train, y_train)
        end = time.time()
        result['time_train'] = end - start

        # Predict train
        start = time.time()
        y_hat = clf.predict(X_train)
        end = time.time()
        result['score_train'] = scoring(y_train.values, y_hat)
        result['time_predict_train'] = end - start

        # Predict test
        start = time.time()
        y_hat = clf.predict(X_test)
        end = time.time()
        result['score_test'] = scoring(y_test.values, y_hat)
        result['time_predict_test'] = end - start

        results.append(result)

    return pd.DataFrame(results)[[
        'description', 'score_train', 'score_test',
        'time_train', 'time_predict_train', 'time_predict_test']]
