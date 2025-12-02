import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import geopandas
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class Preprocessing:

    def __init__(self):
        pass
    
    ### Compression function to save RAM ###

    def compress(df, **kwargs):
        """
        Reduces the size of the DataFrame by 
        1. downcasting numerical columns
        2. remove duplicated values
        """
        
        input_size = df.memory_usage(index=True).sum()/ 1024**2
        print("old dataframe size: ", round(input_size,2), 'MB')

        in_size = df.memory_usage(index=True).sum()
        df.drop(columns=["pr_variable","tas_variable","tasmin_variable","tasmax_variable","rsds_variable", 
                        'soil_co2_crop', 'soil_co2_year', 'soil_co2_lon', 'soil_co2_lat',
                        'tas_crop', 'tas_year', 'tas_lon','tas_lat',
                        'tasmin_crop', 'tasmin_year', 'tasmin_lon', 'tasmin_lat',
                        'tasmax_crop', 'tasmax_year', 'tasmax_lon', 'tasmax_lat',
                        'rsds_crop', 'rsds_year', 'rsds_lon', 'rsds_lat'], inplace=True, axis=1)
        
        rename_map = {
            "pr_crop": "crop",
            "pr_lon": "lon",
            "pr_lat": "lat",
            "soil_co2_texture_class": "texture_class",
            "soil_co2_real_year": "real_year",
            "pr_year": "season_year"
        }
        df.rename(columns=rename_map, inplace=True)
        
        for t in ["float", "integer"]:
            l_cols = list(df.select_dtypes(include=t))

            for col in l_cols:
                df[col] = pd.to_numeric(df[col], downcast=t)

        #dropping duplicates -- can be handled manually with column names
        # df = df.loc[:,~df.apply(lambda x: x.duplicated(),axis=1).all()].copy() --> 
        
        out_size = df.memory_usage(index=True).sum()
        ratio = (1 - round(out_size / in_size, 2)) * 100

        col_to_move = ["crop","lon","lat","texture_class","real_year", "season_year"]
        new_order = col_to_move + [col for col in df.columns if col not in col_to_move]
        df = df.loc[:,new_order]

        print("optimized size by {} %".format(round(ratio,2)))
        print("new DataFrame size: ", round(out_size / 1024**2,2), " MB")

        X = df.drop(columns="yield")
        y = df[["yield"]]

        return X, y
    
    ### Basic feature engineering ###

    def feature_engineering(X: pd.DataFrame)-> pd.DataFrame:
    
        #1 Hydrométrie (total annuel, moyenne, min et max, 30j glissant?) -> 9 features
        pr_columns = [col for col in X.columns if col.startswith('pr_')]

        pr_day_cols = sorted(pr_columns, key=lambda x: int(x.split('_')[-1]))
        rolling_30_days_pr = X[pr_day_cols].T.rolling(window=30, min_periods=30).mean().T.dropna(axis=1)
        rolling_30_days_pr = rolling_30_days_pr.add_prefix('pr_roll30')
        
        mean_pr = X[pr_columns].mean(axis=1).rename('mean_pr')
        sum_pr = X[pr_columns].sum(axis=1).rename('sum_pr')
        min_pr = X[pr_columns].min(axis=1).rename('min_pr')
        max_pr = X[pr_columns].max(axis=1).rename('max_pr')
        
        #2 Températures (min, max, moyenne - annuelles/ glissants? - 10 jours pour le gel?) -> 27 features
        tas_columns = [col for col in X.columns if col.startswith('tas_')]
        mean_tas = X[tas_columns].mean(axis=1).rename('mean_tas')
        min_tas = X[tas_columns].min(axis=1).rename('min_tas')
        max_tas = X[tas_columns].max(axis=1).rename('max_tas')

        #3 Ensoleillement (journalier, moyenne)
        rsds_columns = [col for col in X.columns if col.startswith('rsds_')]
        mean_rsds = X[rsds_columns].mean(axis=1).rename('mean_rsds')
        sum_rsds = X[rsds_columns].sum(axis=1).rename('sum_rsds')
        min_rsds = X[rsds_columns].min(axis=1).rename('min_rsds')
        max_rsds = X[rsds_columns].max(axis=1).rename('max_rsds')

        #4 Découpage Géo
        # - Tropical : [70, 90] ou [-90, -70]
        # - Tempered : [50, 70[ ou ]-70, -50]
        # - Tropical : Sinon (inclut implicitement 0-50)
        
        lat_abs = X['lat'].abs()
        
        conditions = [
            (lat_abs >= 70) & (lat_abs <= 90),
            (lat_abs >= 50) & (lat_abs < 70)
        ]
        
        choices = ['tropical', 'Tempered']
        
        region = pd.Series(np.select(conditions, choices, default='tropical'), 
                        index=X.index, name='region')
        
        # Returning featured df 
        X = pd.concat([mean_pr, sum_pr,min_pr,max_pr, rolling_30_days_pr,
                    mean_tas,min_tas,max_tas,
                    mean_rsds, sum_rsds,min_rsds,max_rsds, region], axis=1)
        return X
    
    
    ### Scaling functions ###
    
    def robust_scaling(X: pd.DataFrame, cutoff_date: int) -> pd.DataFrame:
    
        X_train = X.loc[X.index < cutoff_date]
        X_val = X.loc[X.index >= cutoff_date]
        rb_scaler = RobustScaler()
        X_train_scaled = rb_scaler.fit_transform(X_train)
        X_val_scaled = rb_scaler.transform(X_val)
        return X_train_scaled, X_val_scaled
    
    def standard_scaling(X: pd.DataFrame, cutoff_date: int) -> pd.DataFrame:

        X_train = X.loc[X.index < cutoff_date]
        X_val = X.loc[X.index >= cutoff_date]
        std_scaler = StandardScaler()
        X_train_scaled = std_scaler.fit_transform(X_train)
        X_val_scaled = std_scaler.transform(X_val)
        return X_train_scaled, X_val_scaled
    
    def minmax_scaling(X: pd.DataFrame, cutoff_date: int) -> pd.DataFrame:

        X_train = X.loc[X.index < cutoff_date]
        X_val = X.loc[X.index >= cutoff_date]
        minmax_scaler = MinMaxScaler()
        X_train_scaled = minmax_scaler.fit_transform(X_train)
        X_val_scaled = minmax_scaler.transform(X_val)
        return X_train_scaled, X_val_scaled

    

if __name__ == "__main__":
    preproc = Preprocessing
    wheat_train_datasets = [
    {'file_name': 'pr_', 'path': 'raw_data/pr_wheat_train.parquet'},
    {'file_name': 'soil_co2_', 'path': 'raw_data/soil_co2_wheat_train.parquet'},
    {'file_name': 'tas_', 'path': 'raw_data/tas_wheat_train.parquet'},
    {'file_name': 'tasmin_', 'path': 'raw_data/tasmin_wheat_train.parquet'},
    {'file_name': 'tasmax_', 'path': 'raw_data/tasmax_wheat_train.parquet'},
    {'file_name': 'rsds_', 'path': 'raw_data/rsds_wheat_train.parquet'},
    {'file_name': '', 'path': 'raw_data/train_solutions_wheat.parquet'}
    ]

    # Read and store each wheat train dataset with file name prefix
    wheat_train_dfs = [pd.read_parquet(file['path']).add_prefix(file['file_name']) for file in wheat_train_datasets]

    # Concatenate all wheat train datasets horizontally
    wheat_train_df = pd.concat(wheat_train_dfs, axis=1)
    del wheat_train_dfs

    X, y = preproc.compress(wheat_train_df, verbose=True)
    df = preproc.feature_engineering(X)

    col_to_rb_scale = []
    col_to_std_scale = []
    col_to_minmax_scale = []

    rb_df = X[col_to_rb_scale].copy()
    std_df = X[col_to_std_scale].copy()
    minmax_df = X[col_to_minmax_scale].copy()
    X_train_rb, X_val_rb = preproc.robust_scaling(rb_df)
    X_train_std, X_val_std = preproc.standard_scaling(std_df)
    X_train_minmax, X_val_minmax = preproc.minmax_scaling(minmax_df)

    X_train = pd.concat(X_train_rb, X_train_std, axis=1)
    X_train = pd.concat(X_train, X_train_minmax, axis=1)
    X_val = pd.concat(X_val_rb, X_val_std, axis=1)
    X_val = pd.concat(X_val, X_val_minmax, axis=1)
