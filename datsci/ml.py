"""Tools for running machine learning algorithms
"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2016.03.14
# Last Modified   : 2016.04.18
#
# License         : MIT


import numpy as np
import pandas as pd
import time
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.scorer import check_scoring, get_scorer


class LRRegressor(LogisticRegression):
    """Logistic Regressor that returns probability values intead of class labels

    Useful for use with AdaBoostRegressor

    Example
    -------

    model = ml.LRRegressor(
        penalty='l2', C=1.0, random_state=5
    )
    model_abr = AdaBoostRegressor(
        base_estimator=model, n_estimators=10, learning_rate=1.0,
        loss='linear', random_state=5)

    model_abr.fit(X_train, y_train)
    """
    def predict(self, X):
        return self.predict_proba(X)[:, 1]


def train_predict(descriptions_models,
                  X_train, y_train,
                  X_valid, y_valid,
                  scoring=None):
    """Run preliminary performance analyses of multiple machine learning models.

    Parameters
    ----------
    descriptions_models : Iterable of 2-tuples (str, object)
        Each 2-tuple element contains descriptive text and a model object.
        i.e. [('Model1 info', model1), ('Model2 info', model2), ...]

    X_train : pandas.DataFrame
        Training features data

    y_train : pandas.Series
        Training target data

    X_valid, y_valid : same as X_train, y_train, but used for validation

    scoring : str, callable or None, default=None
        See `scoring` parameter description for
        sklearn.grid_search.GridSearchCV.html

    Returns
    -------
    df_summary : pandas.DataFrame
        Performance summary of all the models
    """

    results = []
    for description, model in descriptions_models:

        scorer = check_scoring(model, scoring=scoring)
        result = {'description': description}

        # Train
        start = time.time()
        model.fit(X_train, y_train)
        result['time_train'] = time.time() - start

        # Predict train
        start = time.time()
        result['score_train'] = scorer(model, X_train, y_train)
        result['time_predict_train'] = time.time() - start

        # Predict validation
        start = time.time()
        result['score_valid'] = scorer(model, X_valid, y_valid)
        result['time_predict_valid'] = time.time() - start

        results.append(result)

    return pd.DataFrame(results)[[
        'description', 'score_train', 'score_valid',
        'time_train', 'time_predict_train', 'time_predict_valid']]


def fine_tune_params(model, X_train, y_train, X_valid, y_valid, param_grid,
                     n_runs=5, n_cv=5, scoring=None, n_jobs=2,
                     gscv_kwargs={}):
    """Fine tune model using multiple runs of sklearn's GridSearchCV, since it
    shuffles the data per run.

    Parameters
    ----------
    model: object
        Machine learning model

    X_train : pandas.DataFrame
        Training features data

    y_train : pandas.Series
        Training target data

    X_valid, y_valid : same as X_train, y_train, but used for validation

    param_grid : dictionary
        Parameter values to use in GridSearchCV

    n_runs : int
        Number of times to run GridSearchCV

    n_cv : int
        GridSearchCV's `cv` parameter

    scoring : str, callable or None, default=None
        See `scoring` parameter description for
        sklearn.grid_search.GridSearchCV.html

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
    scorer = check_scoring(model, scoring=scoring)

    for i in range(n_runs):
        # Print progress
        if i < 3 or i % 10 == 0:
            print("iteration {}".format(i))
            starttime = time.time()

        gs_model = GridSearchCV(
            model, param_grid, cv=n_cv, n_jobs=n_jobs,
            scoring=scoring, **gscv_kwargs
        ).fit(X_train, y_train)

        _score = scorer(gs_model, X_valid, y_valid)
        if best_score is None or best_score < _score:
            best_score = _score
            best_model = gs_model.best_estimator_

        # Print progress
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
                     X_valid, y_valid,
                     cv_nfold=5,
                     early_stopping_rounds=50,
                     missing=np.nan,
                     eval_metric='auc',
                     scoring=None,
                     verbose=True):
    """Fit xgb model with best n_estimators using xgb builtin cv
    Note: This function changes the model's `n_estimators` attribute

    Parameters
    ----------
    model : xgb model object

    X_train : pandas.DataFrame
        Training features data

    y_train : pandas.Series
        Training target data

    X_valid, y_valid : same as X_train, y_train, but used for validation

    cv_nfold : int
        Number of folds in CV

    early_stopping_rounds : int
        Activates early stopping. CV error needs to decrease at least
        every <early_stopping_rounds> round(s) to continue.
        Last entry in evaluation history is the one from best iteration.

    missing : float
        Value in the data which needs to be present as a missing value.

    eval_metric : str
        The metric to be used for validation data while training xgb
        Probably should match `scoring`

    scoring : str, callable or None, default=None
        See `scoring` parameter description for
        sklearn.grid_search.GridSearchCV.html

    verbose : bool
        Print scoring summary to stdout

    Returns
    -------
    best_n_estimators : int
        Number of optimal estimators, or boosting rounds

    train_score : float
        Performance of the best model on training set

    valid_score : float
        Performance of the best model on validation set

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

    n_estimators, train_score, valid_score = cv_fit_xgb_model(
        model, X_train, y_train, X_valid, y_valid, cv_nfold=5,
        early_stopping_rounds=50, scoring='roc_auc', verbose=True
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
        metrics=[eval_metric],
        early_stopping_rounds=early_stopping_rounds,
        show_progress=False)
    best_n_estimators = cv_result.shape[0]
    model.set_params(n_estimators=best_n_estimators)

    # Train model
    model.fit(X_train, y_train, eval_metric=eval_metric)

    scorer = get_scorer(scoring)
    # Predict and score training data
    train_score = scorer(model, X_train, y_train)
    # Predict and score validation data
    valid_score = scorer(model, X_valid, y_valid)

    # Print model report:
    if verbose:
        print("\nModel Report")
        print("best n_estimators: {}".format(best_n_estimators))
        print("Score (Train): %f" % train_score)
        print("Score (Validation) : %f" % valid_score)

    return best_n_estimators, train_score, valid_score
