from sklearn.model_selection import (GridSearchCV, TimeSeriesSplit)
from sklearn.tree import (DecisionTreeRegressor)
from sklearn.ensemble import (GradientBoostingRegressor, AdaBoostRegressor,
                              RandomForestRegressor)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor



def model_selection(X, y, model_dict, params_dict):
    """
    Perform grid-search model selection with time-series cross-validation.

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature matrix used for training.
    y : array-like or pandas.Series
        Target vector.
    model_dict : dict
        Mapping from model name (str) to unfitted estimator instance.
    params_dict : dict
        Mapping from model name (str) to parameter grid (dict or list of dicts)
        for GridSearchCV.

    Returns
    -------
    score_dict : dict
        Mapping model name -> best RMSE (positive float).
    best_params_dict : dict
        Mapping model name -> best parameter dictionary.
    best_estimator_dict : dict
        Mapping model name -> best fitted estimator.
    """
    score_dict = {}
    best_params_dict = {}
    best_estimator_dict = {}

    for name, model in model_dict.items():

        tscv = TimeSeriesSplit(n_splits=5)
        params = params_dict[name]

        grid = GridSearchCV(estimator= model,
                            param_grid=params,
                            cv=tscv,
                            n_jobs=2,
                            pre_dispatch='2*n_jobs',
                            scoring='neg_root_mean_squared_error',
                            verbose=2)

        result = grid.fit(X, y)
        score_dict[name] = -(result.best_score_)
        best_params_dict[name] = result.best_params_
        best_estimator_dict[name] = result.best_estimator_

    return score_dict, best_params_dict, best_estimator_dict


def model_fit(X, y, model):
    """
    Fit a model on the provided data and return the fitted estimator.

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature matrix for training.
    y : array-like or pandas.Series
        Target vector.
    model : estimator
        Estimator implementing fit(X, y).

    Returns
    -------
    estimator
        The fitted estimator (model).
    """
    model.fit(X, y)

    return model


def model_predict(X, model):
    """
    Predict target values using a fitted model.

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature matrix for prediction.
    model : estimator
        Fitted estimator implementing predict(X).

    Returns
    -------
    y_pred : array-like
        Predicted target values.
    """
    y_pred = model.predict(X)

    return y_pred


def model_score(X, y, model):
    """
    Compute the model score on given data using estimator.score.

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature matrix.
    y : array-like or pandas.Series
        True target values.
    model : estimator
        Estimator implementing score(X, y).

    Returns
    -------
    score : float
        Score returned by model.score(X, y).
    """
    score = model.score(X, y)

    return score
