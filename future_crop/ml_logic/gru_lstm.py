from keras import Sequential, layers, Input
from keras.layers import LSTM, GRU, Dense
from keras.callbacks import EarlyStopping
import pandas as pd
from keras.metrics import RootMeanSquaredError
from keras.optimizers import RMSprop, Adam
import numpy as np

'''
These functions are to be used within the future_crop model
and require data to be preprocessed using the inhouse preprocessing functions
'''


def time_columns_selection(X: pd.DataFrame) -> pd.DataFrame:
    '''
    this function takes all the preproc data and then isolates
    the time series related data only.
    It then return the data as a Dataframe.
    '''
    time_columns = [c for c in X.columns if 'pr_' in c] + \
        [c for c in X.columns if 'tas_' in c] + \
        [c for c in X.columns if 'tasmin_' in c] + \
        [c for c in X.columns if 'tasmax_' in c] + \
        [c for c in X.columns if 'rsds_' in c] + \
        ['ID', 'real_year', 'lon', 'lat']

    roll_columns = [c for c in X.columns if 'pr_roll' in c]

    X = X[time_columns]
    X = X.drop(columns = roll_columns)
    return X

def GRU_input_maker(X : pd.DataFrame) -> pd.DataFrame:
    '''
    This function takes the 2D time series dataframe and returns a
    3D GRU compatible dataframe.
    '''

    TIME_STEPS = 240 # Le nombre de pas de temps (de _0 à _239)
    N_FEATURES = 5   # Le nombre de caractéristiques (pr, tas, tasmin, tasmax, rsds)

    # --- 2. Créer la liste ordonnée des colonnes séquentielles ---
    # Regrouper les colonnes par pas de temps (T=0, T=1, ...)
    sequence_cols = []
    feature_names = ['pr', 'tas', 'tasmin', 'tasmax', 'rsds']

    for t in range(TIME_STEPS):
        # Ajouter les colonnes pour le pas de temps 't' dans l'ordre des caractéristiques
        for feat in feature_names:
            col_name = f'{feat}_{t}'
            sequence_cols.append(col_name)

    # --- 3. Extraire et remodeler les données ---

    # a) Extraire les valeurs NumPy
    X_seq_2d = X[sequence_cols].values


    # b) Appliquer le reshape en 3D
    X_train_gru_input = X_seq_2d.reshape(
        X_seq_2d.shape[0], # N_samples (nombre d'échantillons)
        TIME_STEPS,       # Pas de temps (240)
        N_FEATURES        # Caractéristiques (5)
    )
    return X_train_gru_input

def build_lstm_model():
    '''
    This function builds a basdic LSTM model.
    '''
    model = Sequential()

    model.add(LSTM(64, input_shape=((240,5))))  # only temp
    model.add(Dense(1))                      # prédiction de rendement

    model.compile(optimizer=RMSprop(), loss = 'mse', metrics=RootMeanSquaredError(name="rmse"))
    return model

def build_gru_model():
    '''
    This function returns a basic GRU model.
    '''
    model = Sequential()

    model.add(GRU(64, input_shape=((240,5))))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss = 'mse', metrics=RootMeanSquaredError(name="rmse"))
    return model
