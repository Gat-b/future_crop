import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, OneHotEncoder
from pathlib import Path
import gc


class Preprocessing_ml:
    """
    A class for preprocessing data for a machine learning model.

    This class handles loading raw data, compressing it to save memory,
    performing feature engineering, splitting the data into training and
    validation sets, scaling the features, and saving the processed data.

    Attributes:
        raw_data_path (pathlib.Path): The path to the raw data directory.
        processed_data_path (pathlib.Path): The path to the processed data directory.
        crop (str): The type of crop (e.g., 'wheat').
        file (str): The file type (e.g., 'train').
    """

    def __init__(self, raw_data_path='../raw_data', processed_data_path='../processed_data'):

        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

    ### Data loader - only local for now ###

    def load_raw_data(self, crop: str = 'wheat', mode: str = 'train') -> pd.DataFrame:
        """Charge et merge les fichiers parquet bruts."""
        print(f"Loading raw data for {crop} {mode}...")
        crop_train_datasets = [
            {'file_name': 'pr_', 'path': self.raw_data_path / f'pr_{crop}_{mode}.parquet'},
            {'file_name': 'soil_co2_', 'path': self.raw_data_path / f'soil_co2_{crop}_{mode}.parquet'},
            {'file_name': 'tas_', 'path': self.raw_data_path / f'tas_{crop}_{mode}.parquet'},
            {'file_name': 'tasmin_', 'path': self.raw_data_path / f'tasmin_{crop}_{mode}.parquet'},
            {'file_name': 'tasmax_', 'path': self.raw_data_path / f'tasmax_{crop}_{mode}.parquet'},
            {'file_name': 'rsds_', 'path': self.raw_data_path / f'rsds_{crop}_{mode}.parquet'}]

        if mode == 'train':
            crop_train_datasets.append(
                {'file_name': '', 'path': self.raw_data_path / f'{mode}_solutions_{crop}.parquet'})


        dfs = []
        for file in crop_train_datasets:
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
        median_pr = X[pr_columns].median(axis=1).rename('median_pr')
        sum_pr = X[pr_columns].sum(axis=1).rename('sum_pr')
        min_pr = X[pr_columns].min(axis=1).rename('min_pr')
        max_pr = X[pr_columns].max(axis=1).rename('max_pr')

        #2 Températures (min, max, moyenne - annuelles/ glissants? - 10 jours pour le gel?) -> 27 features
        tas_columns = [col for col in X.columns if col.startswith('tas_') and 'tasmin' not in col and 'tasmax' not in col]
        mean_tas = X[tas_columns].mean(axis=1).rename('mean_tas')
        median_tas = X[tas_columns].median(axis=1).rename('median_tas')
        min_tas = X[tas_columns].min(axis=1).rename('min_tas')
        max_tas = X[tas_columns].max(axis=1).rename('max_tas')

        #3 Ensoleillement (journalier, moyenne)
        rsds_columns = [col for col in X.columns if col.startswith('rsds_')]
        mean_rsds = X[rsds_columns].mean(axis=1).rename('mean_rsds')
        median_rsds = X[rsds_columns].median(axis=1).rename('median_rsds')
        sum_rsds = X[rsds_columns].sum(axis=1).rename('sum_rsds')
        min_rsds = X[rsds_columns].min(axis=1).rename('min_rsds')
        max_rsds = X[rsds_columns].max(axis=1).rename('max_rsds')

        #4 tasmin
        tasmin_columns = [col for col in X.columns if col.startswith('tasmin_')]
        mean_tasmin = X[tasmin_columns].mean(axis=1).rename('mean_tasmin')
        median_tasmin = X[tasmin_columns].median(axis=1).rename('median_tasmin')
        sum_tasmin = X[tasmin_columns].sum(axis=1).rename('sum_tasmin')
        min_tasmin = X[tasmin_columns].min(axis=1).rename('min_tasmin')
        max_tasmin = X[tasmin_columns].max(axis=1).rename('max_tasmin')

        #5 tasmax
        tasmax_columns = [col for col in X.columns if col.startswith('tasmax_')]
        mean_tasmax = X[tasmax_columns].mean(axis=1).rename('mean_tasmax')
        median_tasmax = X[tasmax_columns].median(axis=1).rename('median_tasmax')
        sum_tasmax = X[tasmax_columns].sum(axis=1).rename('sum_tasmax')
        min_tasmax = X[tasmax_columns].min(axis=1).rename('min_tasmax')
        max_tasmax = X[tasmax_columns].max(axis=1).rename('max_tasmax')

        #6 Découpage Géo
        # - Polar : [66, 90] ou [-90, -66]
        # - Tempered : [23, 66[ ou ]-66, -23]
        # - Tropical : Sinon (inclut implicitement 0-23)
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

        region_encoded.index = X.index

        #5 Année
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
        X = pd.concat([year, constant, texture, mean_pr,median_pr, sum_pr,min_pr,max_pr, rolling_30_days_pr,
                    mean_tas, median_tas, min_tas, max_tas,
                    mean_rsds, median_rsds, sum_rsds,min_rsds,max_rsds,
                    mean_tasmin, median_tasmin, sum_tasmin,min_tasmin,max_tasmin,
                    mean_tasmax, median_tasmax, sum_tasmax,min_tasmax,max_tasmax,
                    region_encoded, X_raw_features], axis=1)
        return X

    ### premier aggrégation du jeu de données ###

    def process_one_dataset(self, crop, mode):
        """Orchestre load -> compress -> feature eng pour UN fichier."""
        df_raw = self.load_raw_data(crop, mode)
        X_raw, y = self.compress(df_raw)
        del df_raw    # supprimer la variable inutilisé pour optimiser la mémoire
        gc.collect()  #optimiser la RAM --> package python à installer

        X_feat = self.feature_engineering(X_raw)

        # Ré-attacher y si c'est le train, pour garder l'alignement
        if y is not None:
            # Sécurité alignement index
            y = y.loc[X_feat.index]

        return X_feat, y

    ### Scaling ###

    def fit_transform_scaling(self, df_fit: pd.DataFrame, df_transform: pd.DataFrame) -> pd.DataFrame:
        """
        Apprend les scalers sur df_fit (ex: X_train) et applique sur df_transform (ex: X_val ou X_test).
        Renvoie SEULEMENT df_transform scalé.
        """
        print("Scaling data...")
        X_scaled = df_transform.copy()

        # Définition des groupes de colonnes
        cols_pr = [c for c in df_fit.columns if 'pr' in c]
        cols_rsds = [c for c in df_fit.columns if 'rsds' in c]
        cols_tas = [c for c in df_fit.columns if 'tas' in c] # inclut tas, tasmin, tasmax
        cols_geo = ['lat', 'lon']
        cols_co2 = ['soil_co2_co2', 'soil_co2_nitrogen']

        # Scalers
        scalers = {
            'pr': (RobustScaler(), cols_pr),
            'co2': (RobustScaler(), cols_co2),
            'rsds': (MinMaxScaler(), cols_rsds),
            'tas': (StandardScaler(), cols_tas)
        }

        for name, (scaler, cols) in scalers.items():
            if cols:
                # Fit sur le jeu d'entrainement/fit
                scaler.fit(df_fit[cols])
                # Transform sur le jeu cible
                X_scaled[cols] = scaler.transform(df_transform[cols])

        # Traitement spécifique Geo (Cos transformation)
        if all(c in df_fit.columns for c in cols_geo):
             X_scaled[cols_geo] = np.cos(df_transform[cols_geo] / 90)

        return X_scaled

    ### Saving ###

    def save_df(self, df, filename):
        if df is not None:
            path = self.processed_data_path / f"{filename}.csv"
            print(f"Saving {filename} ({df.shape})...")
            df.to_csv(path, index=True)

    ### Handling 2 cases: exploratory and full production cases ###

    def run_exploration(self, crops=['wheat'], cutoff_year=2010):
        """
        Function to handle exploratroy cases on maze or wheat.
        """
        for crop in crops:
            print(f"\n=== Running Exploration for {crop} ===")
            X, y = self.process_one_dataset(crop, 'train')  # c'est ici qu'on ajoute du feature engineering + compression

            # Split Temporel - pour jeu de val
            if 'real_year' not in X.columns:
                 # Fallback si feature engineering a supprimé real_year, à ajuster
                 raise ValueError("real_year missing")

            mask_train = X['real_year'] < cutoff_year
            mask_val = X['real_year'] >= cutoff_year

            X_train_raw, X_val_raw = X[mask_train], X[mask_val]
            y_train, y_val = y[mask_train], y[mask_val]

            # Scaling (On fit sur Train, on transforme Train ET Val)
            X_train_scaled = self.fit_transform_scaling(X_train_raw, X_train_raw)
            X_val_scaled = self.fit_transform_scaling(X_train_raw, X_val_raw)

            # Save
            self.save_df(X_train_scaled, f"X_train_{crop}_explo")
            self.save_df(X_val_scaled, f"X_val_{crop}_explo")
            self.save_df(y_train, f"y_train_{crop}_explo")
            self.save_df(y_val, f"y_val_{crop}_explo")

            # Clean RAM
            del X, y, X_train_raw, X_val_raw
            gc.collect()

    def run_production(self, crops=['wheat', 'maize']):
        """
        fuction to run full production preproc --> before kaggle submission
        """
        for crop in crops:
            print(f"\n=== Running Production for {crop} ===")

            # 1. Process Train Full
            X_train_full, y_train_full = self.process_one_dataset(crop, 'train')

            # 2. Process Test Full
            X_test_full, _ = self.process_one_dataset(crop, 'test')

            # 3. Scaling: Fit sur Train Full, Transform sur Train Full ET Test Full
            X_train_scaled = self.fit_transform_scaling(X_train_full, X_train_full)
            X_test_scaled = self.fit_transform_scaling(X_train_full, X_test_full)

            # 4. Save
            self.save_df(X_train_scaled, f"X_train_{crop}_full")
            self.save_df(y_train_full, f"y_train_{crop}_full")
            self.save_df(X_test_scaled, f"X_test_{crop}_full")

            # Clean RAM
            del X_train_full, X_test_full, y_train_full
            gc.collect()

if __name__ == "__main__":

    preproc = Preprocessing_ml(raw_data_path='raw_data', processed_data_path='processed_data')

    # CAS 1 : Tout générer pour la prod (6 fichiers)
    # preproc.run_production(crops=['wheat', 'maize'])

    # CAS 2 : Juste Wheat pour la prod (3 fichiers)
    # preproc.run_production(crops=['wheat'])

    # CAS 3 : Exploration Wheat (Train/Val split)
    preproc.run_exploration(crops=['wheat'], cutoff_year=2010)

    # CAS 4 : Exploration All crops
    # preproc.run_exploration(crops=['wheat', 'maize'], cutoff_year=2010)
