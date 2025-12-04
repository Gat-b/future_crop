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

# params dict GridSearch
params_dict_grid={
    'DecisionTreeRegressor': {'max_depth': [3, 5, 10, 18],
                            'min_samples_split': [2, 5, 10,],
                            'min_samples_leaf': [1, 2, 5, 10],
                            },
    'GradientBoostingRegressor': {'learning_rate': [0.01, 0.05, 0.1, 0.5],
                                'max_depth': [5, 10, 20],
                                'n_estimators': [10, 30, 50]},
    'AdaBoostRegressor': {'learning_rate': [0.01, 0.05, 0.1, 0.5],
                        'n_estimators': [10, 50, 100],
                        },
    'RandomForestRegressor': {'max_depth': [15, 20],
                            'min_samples_split': [2, 5],
                            'min_samples_leaf': [3, 5],
                            'n_estimators': [100, 150],
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

# params dict RandomizeSearch
params_dict_rando={
    'DecisionTreeRegressor': {'max_depth': [3, 5, 10, 18],
                            'min_samples_split': [2, 5, 10,],
                            'min_samples_leaf': [1, 2, 5, 10],
                            },
    'GradientBoostingRegressor': {'learning_rate': [0.01, 0.05, 0.1, 0.5],
                                'max_depth': [5, 10, 20],
                                'n_estimators': [10, 30, 50]},
    'AdaBoostRegressor': {'learning_rate': [0.01, 0.05, 0.1, 0.5],
                        'n_estimators': [10, 50, 100],
                        },
    'RandomForestRegressor': {'max_depth': [None, 15, 20, 25],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [3, 5, 10],
                            'n_estimators': [50, 100, 200, 300],
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
adri_model_dict = {key: model_dict[key] for key in ['RandomForestRegressor']}
adri_params_dict = {key: params_dict_rando[key] for key in ['RandomForestRegressor']}

simon_model_dict = {}
simon_params_dict = {}

gat_model_dict = {}
gat_params_dict = {}

greg_model_dict = {'XGBRegressor': XGBRegressor(
        tree_method='hist',      # Optimized histogram algorithm
        device='cuda',           # Explicitly use NVIDIA CUDA
        n_jobs=-1
    ),
    'LightGBM': LGBMRegressor(
        device='gpu',            # Enable GPU training
        n_jobs=-1
    ),
    'Catboost': CatBoostRegressor(
        task_type='GPU',         # Crucial for CatBoost
        devices='1',             # Use the first GPU
        verbose=0
    )
}
greg_params_dict = {key: params_dict_grid[key] for key in ['XGBRegressor', 'LightGBM', 'Catboost']}

mat_model_dict = {}
mat_params_dict = {}

# features ml
features_name_ml =['mean_pr', 'median_pr', 'sum_pr', 'min_pr', 'max_pr',
                   'mean_tas', 'median_tas', 'min_tas', 'max_tas',
                   'mean_rsds', 'median_rsds', 'sum_rsds', 'min_rsds', 'max_rsds',
                   'mean_tasmin', 'median_tasmin', 'sum_tasmin', 'min_tasmin', 'max_tasmin',
                   'mean_tasmax', 'median_tasmax', 'sum_tasmax', 'min_tasmax', 'max_tasmax',
                   'soil_co2_co2', 'soil_co2_nitrogen',
                   'lon', 'lat']
