"""Tools for running machine learning algorithms

"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2016.03.14
# Last Modified   : 2016.04.13
#
# License         : MIT


import numpy as np
import pandas as pd
import time
import xgboost as xgb
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


def cv_fit_xgb_model(model,
                     X_train, y_train,
                     X_test, y_test,
                     cv_nfold=5,
                     early_stopping_rounds=50,
                     missing=np.nan,
                     scorer=accuracy_score,
                     verbose=True):
    """Fit xgb model with best n_estimators using xgb builtin cv

    Parameters
    ----------
    model : xgb model object

    X_train : pandas.DataFrame
        Training features data

    y_train : pandas.Series
        Training target data

    X_test, y_test : same as X_train, y_train, but used for testing

    cv_nfold : int
        Number of folds in CV

    early_stopping_rounds : int
        Activates early stopping. CV error needs to decrease at least
        every <early_stopping_rounds> round(s) to continue.
        Last entry in evaluation history is the one from best iteration.

    missing : float
        Value in the data which needs to be present as a missing value.

    scorer : function or method
        Measures performance of a model, takes 2 parameters, y_true and y_hat

    verbose : bool
        Print scoring summary to stdout

    Returns
    -------
    best_n_estimators : int
        Number of optimal estimators, or boosting rounds

    train_score : float
        Performance of the best model on training set

    test_score : float
        Performance of the best model on test set

    Example
    -------

    model = xgb.XGBRegressor(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=1.0,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        max_delta_step=0,
        objective='binary:logistic',
        nthread=4,
        seed=5
    )

    n_estimators, train_score, test_score = cv_fit_xgb_model(
        model, X_train, y_train, X_test, y_test, cv_nfold=5,
        early_stopping_rounds=50, scorer=roc_auc_score, verbose=True
    )
    """
    # Train cv
    xgb_param = model.get_xgb_params()
    dtrain = xgb.DMatrix(X_train.values, label=y_train.values, missing=missing)
    cv_result = xgb.cv(
        xgb_param,
        dtrain,
        num_boost_round=model.get_params()['n_estimators'],
        nfold=cv_nfold,
        metrics=['auc'],
        early_stopping_rounds=early_stopping_rounds,
        show_progress=False)
    best_n_estimators = cv_result.shape[0]
    model.set_params(n_estimators=best_n_estimators)

    # Train model
    model.fit(X_train, y_train, eval_metric='auc')

    # Predict training data
    y_hat_train = model.predict(X_train)

    # Predict test data
    y_hat_test = model.predict(X_test)

    train_score = scorer(y_train, y_hat_train)
    test_score = scorer(y_test,  y_hat_test)

    # Print model report:
    if verbose:
        print("\nModel Report")
        print("best n_estimators: {}".format(best_n_estimators))
        print("Score (Train): %f" % train_score)
        print("Score (Test) : %f" % test_score)

    return best_n_estimators, train_score, test_score
