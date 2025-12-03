from sklearn.model_selection import (GridSearchCV, TimeSeriesSplit)
from sklearn.tree import (DecisionTreeRegressor)
from sklearn.ensemble import (GradientBoostingRegressor, AdaBoostRegressor,
                              RandomForestRegressor)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# model dict
model_dict = {
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'AdaBoostRegressor': AdaBoostRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'XGBRegressor': XGBRegressor(),
    'LightGBM': LGBMRegressor(),
    'Catboost': CatBoostRegressor()
    }

# params dict
params_dict={
    'DecisionTreeRegressor': {'max_depth': [3, 5, 10, 18],
                            'min_samples_split': [2, 5, 10, 20],
                            'min_samples_leaf': [1, 2, 5, 10],
                            },
    'GradientBoostingRegressor': {'learning_rate': [0.01, 0.05, 0.1, 0.5],
                                'max_depth': [3, 5, 10, 20],
                                'n_estimators': [10, 50, 100]},
    'AdaBoostRegressor': {'learning_rate': [0.01, 0.05, 0.1, 0.5],
                        'n_estimators': [10, 50, 100],
                        },
    'RandomForestRegressor': {'max_depth': [3, 5, 10, 20],
                            'min_samples_split': [2, 5, 10, 20],
                            'min_samples_leaf': [1, 2, 5, 10],
                            'n_estimators': [10, 50, 100],
                            },
    'XGBRegressor': {'learning_rate': [0.01, 0.05, 0.1, 0.5],
                    'max_depth': [3, 5, 10, 18],
                    'n_estimators': [10, 50, 100]},
    'LightGBM': {'learning_rate': [0.01, 0.05, 0.1, 0.5],
                    'max_depth': [3, 5, 10, 18],
                    'n_estimators': [10, 50, 100]},
    'Catboost': {'learning_rate': [0.01, 0.05, 0.1, 0.5],
                    'max_depth': [3, 5, 10, 20],
                    'n_estimators': [10, 50, 100]}
    }

# custom dict
adri_model_dict = {key: model_dict[key] for key in ['RandomForestRegressor', 'XGBRegressor', 'LightGBM']}
adri_params_dict = {key: params_dict[key] for key in ['RandomForestRegressor', 'XGBRegressor', 'LightGBM']}

simon_model_dict = {}
simon_params_dict = {}

gat_model_dict = {}
gat_params_dict = {}

greg_model_dict = {key: model_dict[key] for key in ['XGBRegressor', 'LightGBM', 'Catboost']}
greg_params_dict = {key: params_dict[key] for key in ['XGBRegressor', 'LightGBM', 'Catboost']}

mat_model_dict = {}
mat_params_dict = {}

# features ml
features_name_ml =['mean_pr', 'median_pr', 'sum_pr', 'min_pr', 'max_pr',
                   'mean_tas', 'median_tas', 'min_tas', 'max_tas',
                   'mean_rsds', 'median_rsds', 'sum_rsds', 'min_rsds', 'max_rsds',
                   'soil_co2_co2', 'soil_co2_nitrogen',
                   'lon', 'lat']
