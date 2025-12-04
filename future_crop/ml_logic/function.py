from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor



def model_selection_grid(X, y, model_dict, params_dict):
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



def model_selection_randomize(X, y, model_dict, params_dict):
    """
    Effectue une recherche d'hyperparamètres aléatoire (RandomizedSearchCV) pour plusieurs modèles
    en utilisant une validation croisée temporelle.

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Matrice de caractéristiques pour l'entraînement.
    y : array-like or pandas.Series
        Vecteur cible.
    model_dict : dict
        Dictionnaire mappant le nom du modèle (str) à une instance d'estimateur non entraînée.
    params_dict : dict
        Dictionnaire mappant le nom du modèle (str) à la distribution de paramètres
        (param_distributions) utilisée par RandomizedSearchCV.

    Returns
    -------
    score_dict : dict
        Dictionnaire nom de modèle -> meilleur RMSE (valeur positive).
    best_params_dict : dict
        Dictionnaire nom de modèle -> meilleurs paramètres trouvés.
    best_estimator_dict : dict
        Dictionnaire nom de modèle -> meilleur estimateur entraîné.

    Notes
    -----
    - RandomizedSearchCV est configuré avec n_iter=50, scoring='neg_root_mean_squared_error',
      cv=TimeSeriesSplit(n_splits=5) et n_jobs=-1.
    - Le score renvoyé par RandomizedSearchCV est converti en RMSE positif en inversant le signe.
    """
    score_dict = {}
    best_params_dict = {}
    best_estimator_dict = {}

    for name, model in model_dict.items():

        tscv = TimeSeriesSplit(n_splits=5)
        params = params_dict[name]

        random = RandomizedSearchCV(model,
                                    param_distributions=params,
                                    n_iter=50,
                                    scoring='neg_root_mean_squared_error',
                                    n_jobs=-1,
                                    cv=tscv,
                                    verbose=2)

        result = random.fit(X, y)
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
