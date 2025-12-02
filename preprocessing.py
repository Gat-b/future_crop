import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, OneHotEncoder
from pathlib import Path


class Preprocessing:

    def __init__(self, raw_data_path='raw_data', processed_data_path='processed_data', crop='wheat'):
        
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.crop = crop
        
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
    
    ### Data loader - only local for now ###
    
    def load_raw_data(self) -> pd.DataFrame:
        """Charge et merge les fichiers parquet bruts."""
        print("Loading raw data...")
        wheat_train_datasets = [
            {'file_name': 'pr_', 'path': self.raw_data_path / f'pr_{self.crop}_train.parquet'},
            {'file_name': 'soil_co2_', 'path': self.raw_data_path / f'soil_co2_{self.crop}_train.parquet'},
            {'file_name': 'tas_', 'path': self.raw_data_path / f'tas_{self.crop}_train.parquet'},
            {'file_name': 'tasmin_', 'path': self.raw_data_path / f'tasmin_{self.crop}_train.parquet'},
            {'file_name': 'tasmax_', 'path': self.raw_data_path / f'tasmax_{self.crop}_train.parquet'},
            {'file_name': 'rsds_', 'path': self.raw_data_path / f'rsds_{self.crop}_train.parquet'},
            {'file_name': '', 'path': self.raw_data_path / f'train_solutions_{self.crop}.parquet'}
        ]
        
        dfs = []
        for file in wheat_train_datasets:
            # Vérification basique si le fichier existe
            if not file['path'].exists():
                raise FileNotFoundError(f"Fichier manquant : {file['path']}")
            dfs.append(pd.read_parquet(file['path']).add_prefix(file['file_name']))
        
        print("data_loaded !")    
        return pd.concat(dfs, axis=1)
     
    ### Compression function to save RAM ###

    def compress(self, df: pd.DataFrame) -> pd.DataFrame:
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

        if "yield" in df.columns:
            X = df.drop(columns="yield")
            y = df[["yield"]]
            return X, y
        return df, None
    
    ### Basic feature engineering ###

    def feature_engineering(self, X: pd.DataFrame)-> pd.DataFrame:
    
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
        # - Polar : [70, 90] ou [-90, -70]
        # - Tempered : [50, 70[ ou ]-70, -50]
        # - Tropical : Sinon (inclut implicitement 0-50)
        lat_abs = X['lat'].abs()
        
        conditions = [
            (lat_abs >= 66) & (lat_abs <= 90),
            (lat_abs >= 23) & (lat_abs < 66)
        ]
        
        choices = ['Polar', 'Tempered']
        
        region = pd.Series(np.select(conditions, choices, default='Tropical'), 
                        index=X.index, name='region')
        
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        region_encoded = pd.DataFrame(ohe.fit_transform(region.values.reshape(-1, 1)), 
                                      columns=ohe.get_feature_names_out(['region']))

        #5 Année 
        region_encoded.index = X.index
        year = X['real_year']

        #6 texture sol
        
        texture_class = X[['texture_class']]
        texture = pd.DataFrame(ohe.fit_transform(texture_class), 
                               columns=ohe.get_feature_names_out(['texture_class']))
        texture.index = X.index

        #7 Features non modifiées
        constant = X[['lon', 'lat', 'season_year']]

        #8 tout le reste
        X_raw_features = X.iloc[:, 6:]

        # Returning featured df 
        X = pd.concat([year, constant, texture, mean_pr, sum_pr,min_pr,max_pr, rolling_30_days_pr,
                    mean_tas,min_tas,max_tas,
                    mean_rsds, sum_rsds,min_rsds,max_rsds, region_encoded, X_raw_features], axis=1)
        return X
    
    ### Train/val split ###

    def split_data(self, X: pd.DataFrame, y: pd.DataFrame, cutoff_year=2010) -> tuple:
        """
        Divide data based on a cutoff year.
        """
              
        if 'real_year' not in X.columns:
             raise ValueError("'real_year' column is required for splitting the time searies data")

        print(f"Splitting data with cutoff year: {cutoff_year}")
        mask_train = X['real_year'] < cutoff_year
        print(mask_train)
        mask_val = X['real_year'] >= cutoff_year
        print(mask_val)

        X_train, X_val = X[mask_train], X[mask_val]
        y_train, y_val = y[mask_train], y[mask_val]

        return X_train, X_val, y_train, y_val
    
    ### Scaling functions ###
    
    def custom_scaling(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> tuple:
        """
        Applies different scalers to different column groups:
        - RobustScaler -> 'pr' columns (precipitation)
        - MinMaxScaler -> 'rsds' columns (solar radiation)
        - StandardScaler -> 'tas', 'tasmin', 'tasmax' columns (temperature)
        - Passthrough -> others (lat, lon, encoded features, etc.)
        """
        print("Applying custom scaling...")
        
        # 1. Identify columns by group

        cols_pr = [c for c in X_train.columns if 'pr' in c]
        cols_rsds = [c for c in X_train.columns if 'rsds' in c]
        cols_tas = [c for c in X_train.columns if 'tas' in c]

        col_lat = ['lat']
        
        # All other columns are kept as is
        cols_passthrough = [c for c in X_train.columns if c not in cols_pr + cols_rsds + cols_tas + col_lat]

        # 2. Initialize Scalers
        scaler_pr = RobustScaler()
        scaler_rsds = MinMaxScaler()
        scaler_tas = StandardScaler()

        # 3. Fit & Transform (returns Numpy arrays, so we rebuild DataFrames)
        
        # robust_scaling_pr
        if cols_pr:
            train_pr = pd.DataFrame(scaler_pr.fit_transform(X_train[cols_pr]), 
                                    columns=cols_pr, index=X_train.index)
            val_pr = pd.DataFrame(scaler_pr.transform(X_val[cols_pr]), 
                                  columns=cols_pr, index=X_val.index)
        else:
            train_pr, val_pr = pd.DataFrame(), pd.DataFrame()  #le bloc Else pour éviter tout plantage si pas de donnée pr --> crée un df vide que le .concat va ignorer.

        # minmax scaling rsds
        if cols_rsds:
            train_rsds = pd.DataFrame(scaler_rsds.fit_transform(X_train[cols_rsds]), 
                                      columns=cols_rsds, index=X_train.index)
            val_rsds = pd.DataFrame(scaler_rsds.transform(X_val[cols_rsds]), 
                                    columns=cols_rsds, index=X_val.index)
        else:
            train_rsds, val_rsds = pd.DataFrame(), pd.DataFrame()

        # std sclaing temperature data
        if cols_tas:
            train_tas = pd.DataFrame(scaler_tas.fit_transform(X_train[cols_tas]), 
                                     columns=cols_tas, index=X_train.index)
            val_tas = pd.DataFrame(scaler_tas.transform(X_val[cols_tas]), 
                                   columns=cols_tas, index=X_val.index)
        else:
            train_tas, val_tas = pd.DataFrame(), pd.DataFrame()

        if col_lat:
            train_lat = pd.DataFrame(np.cos(X_train[col_lat]/90), columns=col_lat, index=X_train.index)
            val_lat = pd.DataFrame(np.cos(X_val[col_lat]/90), columns=col_lat, index=X_train.index)
        else:
            train_lat, val_lat = pd.DataFrame(), pd.DataFrame()

        # left it as is
        train_pass = X_train[cols_passthrough]
        val_pass = X_val[cols_passthrough]

        # 4. Concatenate back
        X_train_scaled = pd.concat([train_pass, train_pr, train_rsds, train_tas, train_lat], axis=1)
        X_val_scaled = pd.concat([val_pass, val_pr, val_rsds, val_tas, val_lat], axis=1)
        
        # Ensure same column order as input df
        X_train_scaled = X_train_scaled[X_train.columns]
        X_val_scaled = X_val_scaled[X_val.columns]

        return X_train_scaled, X_val_scaled

    ### Saving and checking data ###

    def check_processed_files(self, prefix="train") -> bool:
        """check if files already exists and processed"""
        files = ['X_train.csv', 'X_val.csv', 'y_train.csv', 'y_val.csv']
        return all((self.processed_data_path / f"{prefix}_{f}").exists() for f in files)
    
    def save_data(self, X_train, X_val, y_train, y_val):
        """Export in CSV on a local dir"""
        print(f"Saving processed data in {self.processed_data_path}...")
        X_train.to_csv(self.processed_data_path / "X_train.csv", index=False)
        print (f"X_train saved! and is of shape: {X_train.shape}")
        X_val.to_csv(self.processed_data_path / "X_val.csv", index=False)
        print (f"X_val saved! and is of shape: {X_val.shape}")
        y_train.to_csv(self.processed_data_path / "y_train.csv", index=False)
        print (f"y_train saved! and is of shape: {y_train.shape}")
        y_val.to_csv(self.processed_data_path / "y_val.csv", index=False)
        print (f"y_val saved! and is of shape: {y_val.shape}")
        print("Done")

    ### Running it all ###

    def run_pipeline(self, cutoff_year=2010, force_reload=False):
        """
        Running the entire preprocessing functions.
        If iles already exists then preprocessing is skipped unless forced. (force_reload=True).
        """

        if self.check_processed_files() and not force_reload:
            print("Preproc already done.")
            return

        # 1. Load
        df_raw = self.load_raw_data()
        
        # 2. Compress & Split features/target
        X_raw, y = self.compress(df_raw)
        
        # 3. Feature Engineering
        X_features = self.feature_engineering(X_raw)
                
        # 4. Split
        X_train, X_val, y_train, y_val = self.split_data(X_features, y, cutoff_year=cutoff_year)
        
        # 5. Scaling 
        X_train, X_val = self.custom_scaling(X_train, X_val)
                
        # 6. Save
        self.save_data(X_train, X_val, y_train, y_val)


if __name__ == "__main__":
    
    preproc = Preprocessing(raw_data_path='raw_data', processed_data_path='processed_data', crop='wheat')
    
    preproc.run_pipeline(cutoff_year=2010, force_reload=True)
