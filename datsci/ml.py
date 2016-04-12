"""Tools for running machine learning algorithms

"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2016.03.14
# Last Modified   : 2016.04.12
#
# License         : MIT


import pandas as pd
import time
from sklearn.grid_search import GridSearchCV as GSCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc


def train_predict(descriptions_clfs,
                  X_train, y_train,
                  X_test, y_test,
                  scorer=accuracy_score):
    """Run preliminary performance analyses of multiple machine learning models.

    Parameters
    ----------
    descriptions_clfs : Iterable of 2-tuples (str, object)
        Each 2-tuple element contains descriptive text and a classifier object.
        i.e. [('Classifier1 info', clf1), ('Classifier2 info', clf2), ...]

    X_train : pandas.DataFrame
        Training features data

    y_train : pandas.Series
        Training target data

    X_test, y_test : same as X_train, y_train, but used for testing

    scorer : function or method
        Measures performance of a model, takes 2 parameters, y_true and y_hat

    Returns
    -------
    df_summary : pandas.DataFrame
        Performance summary of all the models
    """
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
        result['score_train'] = scorer(y_train.values, y_hat)
        result['time_predict_train'] = end - start

        # Predict test
        start = time.time()
        y_hat = clf.predict(X_test)
        end = time.time()
        result['score_test'] = scorer(y_test.values, y_hat)
        result['time_predict_test'] = end - start

        results.append(result)

    return pd.DataFrame(results)[[
        'description', 'score_train', 'score_test',
        'time_train', 'time_predict_train', 'time_predict_test']]


def fine_tune_params(clf, X_train, y_train, X_test, y_test, param_grid,
                     n_runs=5, n_cv=5, scorer=accuracy_score, n_jobs=2,
                     gscv_kwargs={}):
    """Fine tune model using multiple runs of sklearn's GridSearchCV, since it
    shuffles the data per run.

    Parameters
    ----------
    clf: object
        Machine learning model

    X_train : pandas.DataFrame
        Training features data

    y_train : pandas.Series
        Training target data

    X_test, y_test : same as X_train, y_train, but used for testing

    param_grid : dictionary
        Parameter values to use in GridSearchCV

    n_runs : int
        Number of times to run GridSearchCV

    n_cv : int
        GridSearchCV's `cv` parameter

    scorer : function or method
        Measures performance of a model, takes 2 parameters, y_true and y_hat

    n_jobs : int
        GridSearchCV's `n_jobs` parameter

    gscv_kwargs : dict
        Keyword arguments to be passed into GridSearchCV

    Returns
    -------
    best_score: float
        Best(max) score returned by scorer

    best_model: object
        Model object corresponding to best_score
    """
    best_score = None
    best_model = None
    for i in range(n_runs):
        if i < 3 or i % 10 == 0:
            print("iteration {}".format(i))
            starttime = time.time()
        gs_clf = GSCV(clf, param_grid, cv=n_cv, n_jobs=n_jobs,
                      scoring=make_scorer(scorer), **gscv_kwargs)
        gs_clf.fit(X_train, y_train)
        _score = scorer(y_test, gs_clf.predict(X_test))
        if best_score is None or best_score < _score:
            best_score = _score
            best_model = gs_clf.best_estimator_
        if i < 3 or i % 10 == 0:
            runtime = time.time() - starttime
            print("Each iteration time(secs): {:.3f}".format(runtime))

    return best_score, best_model


def plot_roc(y_true, y_hat):
    """Plot ROC curve
    """
    tpr_label = "True Positive Rate"
    fpr_label = "False Positive Rate"
    fpr, tpr, thresholds = roc_curve(y_true, y_hat)
    _area = auc(fpr, tpr)
    title = 'Receiver Operating Characteristic (AUROC={})'.format(_area)
    ax = pd.DataFrame({
        fpr_label: fpr,
        tpr_label: tpr
    }).set_index(fpr_label).plot(
        kind='line',
        title=title,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        legend=False
    )
    ax.set_ylabel(tpr_label)
    return ax
