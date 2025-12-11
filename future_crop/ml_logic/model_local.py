# ------------------------
# Import
# ------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from keras.layers import Input, Conv1D, LSTM, Bidirectional, Dropout, Dense, TimeDistributed
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import RootMeanSquaredError
from keras import backend as K

from tqdm import tqdm
from sklearn.neighbors import BallTree
from sklearn.metrics import mean_squared_error

from future_crop.ml_logic.gru_lstm import make_temporal_features, time_columns_selection_orig

# ------------------------
# Preproc
# ------------------------

def matrice_adj(X, n_neighbors=5):
    """
    Retourne :
        - coord : DataFrame unique (lat, lon)
        - A : matrice d'adjacence pondérée selon distance haversine
    """

    # Extraire coordonnées uniques
    coord = X[['lat_orig', 'lon_orig']].drop_duplicates().reset_index(drop=True)
    print(f"Nombre de noeuds uniques : {len(coord)}")

    # Conversion en radians pour BallTree
    coord_rad = np.radians(coord[['lat_orig', 'lon_orig']].values)

    # Construire BallTree pour haversine
    tree = BallTree(coord_rad, metric='haversine')

    # Rechercher voisins
    distances, indices = tree.query(coord_rad, k=n_neighbors+1)
    # distances en radians ; indice 0 = soi-même

    # Convertir en km (rayon terrestre)
    distances_km = distances * 6371

    # Construire matrice d'adjacence pondérée
    N = len(coord)
    A = np.zeros((N, N))

    for i in range(N):
        for d, j in zip(distances_km[i][1:], indices[i][1:]):
            A[i, j] = 1 / (d + 1e-6)

    return coord, A

def impute_neighbors(tensor, A):
    '''
    Docstring pour impute_neighbors

    :param X_tensor: Description
    :param A: Description

    return padding de X_tensor avec la moyenne de ses 5 voisins
    '''
    X_filled = tensor.copy()
    batch_size, nodes = tensor.shape[0], tensor.shape[1]
    for b in range(batch_size):
        for n in range(nodes):
            mask_nan = np.isnan(tensor[b, n])
            if np.any(mask_nan):
                neighbors = np.where(A[n] > 0)[0]
                if len(neighbors) > 0:
                    mean_neighbor = np.nanmean(tensor[b, neighbors], axis=0)
                    X_filled[b, n][mask_nan] = mean_neighbor[mask_nan]
                else:
                    X_filled[b, n][mask_nan] = 0

    # Reconversion en TensorFlow
    Output_tensor = tf.convert_to_tensor(X_filled, dtype=tensor.dtype)

    return Output_tensor

def preproc_nodes(X_bef, y_bef, coord, A, nb_features=7, test=False):
    '''
    Docstring pour preproc_gcn_X

    :param X: X_train or X_val
    :param y: y_train or y_val
    :param features_num: 5
    :param coord: pd.df coord (lat, lon) unique
    '''
    ### Copy
    X = X_bef.copy()
    y = y_bef.copy()

    ### Params
    years = sorted(X['real_year'].unique())
    nb_years = len(years)
    nb_nodes = len(coord)

    #### X PREPROCESS
    ### Set index
    # X_set = X.set_index('real_year')
    X_set = make_temporal_features(X)
    X_time = time_columns_selection_orig(X_set)

    #### Y PREPROCESS
    # Objectif : relier les données y avec la lat et la lon
    X_coord = X[['lat_orig', 'lon_orig', 'real_year']]
    # X_coord.rename(columns={"Unnamed: 0": "ID"}, inplace=True)
    y_set = X_coord.merge(y, left_index=True, right_index=True)


    ### Ajout des lat et long manquantes
    # 1. Initalisation avec des 0
    X_tensor = np.full((nb_years, nb_nodes, nb_features, 240), 0, dtype=float)
    y_tensor = np.full((nb_years, nb_nodes, 1), 0, dtype=float)
    id = np.full((nb_years, nb_nodes, 1), 0, dtype=float)

    # 2. Création d'un dico pour ajouter l'indice de la coordonnées > Permet davoir toujours le même ordre pour la matrice A
    coord_index = { tuple(c): i for i, c in enumerate(coord[['lat_orig','lon_orig']].itertuples(index=False)) }

    # 3. Création du tenseur final
    for id_year, year in enumerate(years) : # Boucle sur les années

        X_year = X_time[X_time['real_year'] == year]
        y_year = y_set[y_set['real_year'] == year]

        for x_node in X_year.itertuples(index=False):# Boucle sur les noeuds géographiques pour X
            # Trouver le noeud correspondant dans les nodes uniques
            lat = getattr(x_node, 'lat_orig')
            lon = getattr(x_node, 'lon_orig')
            node_id = coord_index[(lat, lon)]

            #Si coord pas présent dans le X_val
            if (lat, lon) not in coord_index:
                continue

            # Extraction de la série temporelle (240 jours, features_num)
            X_series = np.array(x_node[:-5]).reshape(nb_features, 240)

            # Injection dans le tenseur final
            X_tensor[id_year, node_id] = X_series

        for y_node in y_year.itertuples(index=False):# Boucle sur les noeuds géographiques pour y
            #print(y_node)
            # Trouver le noeud correspondant dans les nodes uniques
            lat = getattr(y_node, 'lat_orig')
            lon = getattr(y_node, 'lon_orig')
            real_year = getattr(y_node, 'real_year')

            node_id = coord_index[(lat, lon)]

            # Extraction du rendement
            y_value = y_node[-1]
            y_tensor[id_year, node_id] = y_value

            #Injection dans id de l'ID
            ### On a perdu l'ID de X dans le prepro
            if test == True :
                index = X.loc[(X['lat_orig'] == lat) &
                                            (X['lon_orig'] == lon) &
                                            (X['real_year'] == real_year),
                                            'ID'].values[0]

                id[id_year, node_id] = y.loc[index, 'ID']

            else :
                id[id_year, node_id] = X.loc[(X['lat_orig'] == lat) &
                                            (X['lon_orig'] == lon) &
                                            (X['real_year'] == real_year),
                                            'ID'].values[0]

    ## Ajout des valeurs manquantes par les moyennes des voisins
    X_tensor = impute_neighbors(X_tensor, A)
    y_tensor = impute_neighbors(y_tensor, A)

    #X_tensor = tf.transpose(X_tensor, perm=(0,1,3,2))  # → (years, nodes, 240, features)

    print("Shape de X_tensor:", X_tensor.shape)
    print("Shape de y_tensor:", y_tensor.shape)

    return X_tensor, y_tensor, id

def preproc_nodes_x(X_bef, coord, A, nb_features=7, test=False):
    """
    Prétraite les données pour un modèle local en renvoyant le tenseur X et l'ID associé.

    :param X_bef: pd.DataFrame, données avec colonnes temporelles + 'lat_orig', 'lon_orig', 'real_year', 'Unnamed: 0'
    :param coord: pd.DataFrame, coordonnées uniques des nodes avec colonnes ['lat_orig','lon_orig']
    :param A: np.array, matrice d'adjacence
    :param nb_features: int, nombre de features
    :param test: bool, si True on garde la correspondance avec ID test
    :return:
        - X_tensor: np.array, shape (nb_years, nb_nodes, nb_features, 240)
        - id_tensor: np.array, shape (nb_years, nb_nodes, 1), ID correspondant
    """

    X = X_bef.copy()

    # --- Paramètres ---
    years = sorted(X['real_year'].unique())
    nb_years = len(years)
    nb_nodes = len(coord)

    # --- Préprocessing temporel ---
    print('Préprocessing temporelle...')
    X_set = make_temporal_features(X)
    X_time = time_columns_selection_orig(X_set)

    # --- Initialisation des tenseurs ---
    X_tensor = np.zeros((nb_years, nb_nodes, nb_features, 240), dtype=float)
    id_tensor = np.zeros((nb_years, nb_nodes, 1), dtype=float)

    # --- Mapping coord -> node index ---
    coord_index = { tuple(c): i for i, c in enumerate(coord[['lat_orig','lon_orig']].itertuples(index=False)) }

    # --- Remplissage des tenseurs ---
    print('Remplissage des tenseurs')
    for id_year, year in enumerate(years):
        X_year = X_time[X_time['real_year'] == year]

        for x_node in X_year.itertuples(index=False):
            lat = getattr(x_node, 'lat_orig')
            lon = getattr(x_node, 'lon_orig')

            # Vérifie que le node existe dans coord
            if (lat, lon) not in coord_index:
                continue
            node_id = coord_index[(lat, lon)]

            # Extraction des features temporelles
            X_series = np.array(x_node[:-5]).reshape(nb_features, 240)
            X_tensor[id_year, node_id] = X_series

            # ID correspondant
            index_row = X.loc[
                (X['lat_orig'] == lat) &
                (X['lon_orig'] == lon) &
                (X['real_year'] == year),
                'ID'
            ].values[0]

            id_tensor[id_year, node_id] = index_row

    # --- Imputation des valeurs manquantes par les voisins ---
    X_tensor = impute_neighbors(X_tensor, A)

    print("Shape de X_tensor:", X_tensor.shape)
    print("Shape de id_tensor:", id_tensor.shape)

    return X_tensor, id_tensor




# ------------------------
# Identification des voisins
# ------------------------
def get_neighbors_idx(A, n_neighbors=5):
    neighbors_idx = {}
    for i in range(A.shape[0]):
        row = np.array(A[i]).flatten()
        sorted_idx = np.argsort(-row)
        top_k = [int(idx) for idx in sorted_idx if idx != i][:n_neighbors]
        neighbors_idx[i] = top_k
    return neighbors_idx

# ------------------------
# Modèle Conv1D → LSTM
# ------------------------

def rmse(y_true, y_pred):
    """
    RMSE en TensorFlow pur (compatible Keras 3+)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def rmse_df(df_true, df_pred):
    # Fusionner les data_frame
    merge_df = df_true.merge(df_pred, right_index=True, left_index=True, how='left')

    # Extraction des valeurs
    y_true = merge_df['yield'].astype(float).to_numpy()
    y_pred = merge_df['yield_pred'].astype(float).to_numpy()

    # Calcul RMSE
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def rmse_tf(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def lstm_model(input_shape):

    X_in = Input(shape=input_shape)

    x = Bidirectional(LSTM(64, return_sequences=False))(X_in)
    x = Dropout(0.3)(x)

    out = Dense(1, activation='linear')(x)

    model = Model(X_in, out)

    # --- Compile ---
    model.compile(loss='mse',
                optimizer='adam',
                metrics=[rmse])

    return model

# ------------------------
# Entraînement par node avec ses voisins + validation
# ------------------------

def train_local_models(X_tensor_train, y_tensor_train, X_tensor_val, y_tensor_val, A, n_neighbors=5, epochs=10, batch_size=2):
    n_years, n_nodes, timesteps, n_features = X_tensor_train.shape
    neighbors_idx = get_neighbors_idx(A, n_neighbors)
    models = []

    for node_id in range(n_nodes):
        print(f"Avancement: {node_id + 1} / {n_nodes}")
        # Node + k voisins
        selected_ids = [node_id] + neighbors_idx[node_id]

        # Extraire node + voisins et fusionner features
        X_train_node = tf.gather(X_tensor_train, indices=selected_ids, axis=1)  # (years, k+1, timesteps, features)
        X_train_node = tf.reshape(X_train_node, (n_years, timesteps, (n_neighbors+1)*n_features))
        y_train_node = y_tensor_train[:, node_id, :]  # (years, 1)

        X_val_node = tf.gather(X_tensor_val, indices=selected_ids, axis=1)
        X_val_node = tf.reshape(X_val_node, (X_val_node.shape[0], timesteps, (n_neighbors+1)*n_features))
        y_val_node = y_tensor_val[:, node_id, :]

        # --- Callbacks ---
        es = EarlyStopping(
                    patience=10,
                    restore_best_weights=True,
                    monitor='val_rmse',
                    mode='min'
                )


        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=2, min_lr=1e-6)

        # Construire modèle
        model = lstm_model(input_shape=(timesteps, (n_neighbors+1)*n_features))

        model.fit(
            X_train_node, y_train_node,
            validation_data=(X_val_node, y_val_node),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[reduce_lr, es]
        )

        models.append(model)

    return models

def train_local_models_batched(
        X_tensor_train, y_tensor_train,
        X_tensor_val, y_tensor_val,
        A, n_neighbors=5,
        batch_nodes=16,
        epochs=10,
        batch_size=2):
    """
    Entraîne les modèles locaux par batch de nodes.
    Chaque batch partage un modèle, entraîné sur (B * years) exemples.
    """
    n_years, n_nodes, timesteps, n_features = X_tensor_train.shape
    neighbors_idx = get_neighbors_idx(A, n_neighbors)
    models = [None] * n_nodes

    print(f"\n### Entraînement batché : {batch_nodes} modèles en parallèle ###\n")

    for start in tqdm(range(0, n_nodes, batch_nodes), desc="Training"):
        end = min(start + batch_nodes, n_nodes)
        batch_ids = list(range(start, end))

        # Préparer batchs
        X_train_batch, y_train_batch = [], []
        X_val_batch, y_val_batch = [], []

        for node_id in batch_ids:
            selected_ids = [node_id] + neighbors_idx[node_id]

            Xn_train = tf.gather(X_tensor_train, selected_ids, axis=1)
            Xn_train = tf.reshape(Xn_train, (n_years, timesteps, (n_neighbors + 1) * n_features))
            yn_train = y_tensor_train[:, node_id, :]

            Xn_val = tf.gather(X_tensor_val, selected_ids, axis=1)
            Xn_val = tf.reshape(Xn_val, (Xn_val.shape[0], timesteps, (n_neighbors + 1) * n_features))
            yn_val = y_tensor_val[:, node_id, :]

            X_train_batch.append(Xn_train)
            y_train_batch.append(yn_train)
            X_val_batch.append(Xn_val)
            y_val_batch.append(yn_val)

        # Convertir en tensors batchés
        B = len(batch_ids)
        X_train_flat = tf.reshape(tf.stack(X_train_batch), (B * n_years, timesteps, (n_neighbors + 1) * n_features))
        y_train_flat = tf.reshape(tf.stack(y_train_batch), (B * n_years, 1))
        X_val_flat = tf.reshape(tf.stack(X_val_batch), (B * X_val_batch[0].shape[0], timesteps, (n_neighbors + 1) * n_features))
        y_val_flat = tf.reshape(tf.stack(y_val_batch), (B * y_val_batch[0].shape[0], 1))

        # Callbacks
        es = EarlyStopping(
                        patience=5,
                        restore_best_weights=True,
                        monitor='val_rmse',
                        mode='min'
                    )
        rl = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3)

        # Modèle partagé pour ce batch
        model = lstm_model((timesteps, (n_neighbors + 1) * n_features))
        model.fit(
            X_train_flat, y_train_flat,
            validation_data=(X_val_flat, y_val_flat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es, rl],
            verbose=0
        )

        # Enregistrer le modèle pour chaque node du batch
        for node_id in batch_ids:
            models[node_id] = model

    print("\n🎉 Training terminé !")
    return models

def train_local_models_batched_all(
        X_tensor_train, y_tensor_train,
        A, n_neighbors=5,
        batch_nodes=16,
        epochs=10,
        batch_size=2):
    """
    Entraîne les modèles locaux par batch de nodes.
    Chaque batch partage un modèle, entraîné sur (B * years) exemples.
    """
    n_years, n_nodes, timesteps, n_features = X_tensor_train.shape
    neighbors_idx = get_neighbors_idx(A, n_neighbors)
    models = [None] * n_nodes

    print(f"\n### Entraînement batché : {batch_nodes} modèles en parallèle ###\n")

    for start in tqdm(range(0, n_nodes, batch_nodes), desc="Training"):
        end = min(start + batch_nodes, n_nodes)
        batch_ids = list(range(start, end))

        # Préparer batchs
        X_train_batch, y_train_batch = [], []
        #X_val_batch, y_val_batch = [], []

        for node_id in batch_ids:
            selected_ids = [node_id] + neighbors_idx[node_id]

            Xn_train = tf.gather(X_tensor_train, selected_ids, axis=1)
            Xn_train = tf.reshape(Xn_train, (n_years, timesteps, (n_neighbors + 1) * n_features))
            yn_train = y_tensor_train[:, node_id, :]

            #Xn_val = tf.gather(X_tensor_val, selected_ids, axis=1)
            #Xn_val = tf.reshape(Xn_val, (Xn_val.shape[0], timesteps, (n_neighbors + 1) * n_features))
            #yn_val = y_tensor_val[:, node_id, :]

            X_train_batch.append(Xn_train)
            y_train_batch.append(yn_train)
            #X_val_batch.append(Xn_val)
            #y_val_batch.append(yn_val)

        # Convertir en tensors batchés
        B = len(batch_ids)
        X_train_flat = tf.reshape(tf.stack(X_train_batch), (B * n_years, timesteps, (n_neighbors + 1) * n_features))
        y_train_flat = tf.reshape(tf.stack(y_train_batch), (B * n_years, 1))
        #X_val_flat = tf.reshape(tf.stack(X_val_batch), (B * X_val_batch[0].shape[0], timesteps, (n_neighbors + 1) * n_features))
        #y_val_flat = tf.reshape(tf.stack(y_val_batch), (B * y_val_batch[0].shape[0], 1))

        # Callbacks
        es = EarlyStopping(
                        patience=5,
                        restore_best_weights=True,
                        monitor='val_rmse',
                        mode='min'
                    )

        rl = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3)

        # Modèle partagé pour ce batch
        model = lstm_model((timesteps, (n_neighbors + 1) * n_features))
        model.fit(
            X_train_flat, y_train_flat,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es, rl],
            shuffle=False,
            verbose=0
        )

        # Enregistrer le modèle pour chaque node du batch
        for node_id in batch_ids:
            models[node_id] = model

    print("\n🎉 Training terminé !")
    return models



# ------------------------
# Evaluation des models
# ------------------------

def evaluate_local_models(
    models,
    X_test_tensor,
    y_test_tensor,
    y_test,
    id_test,
    A,
    n_neighbors=5
):
    """
    Évalue les modèles locaux en utilisant X_test_tensor, y_test_tensor et id_test.

    Paramètres
    ----------
    models : liste de modèles locaux
    X_test_tensor : tf.Tensor shape (years, nodes, timesteps, features)
    y_test_tensor : tf.Tensor shape (years, nodes, 1)
    id_test : np.array shape (years, nodes, 1)  → contient l'ID réel ('Unnamed: 0')
    A : matrice d'adjacence
    n_neighbors : nombre de voisins utilisés

    Retour
    ------
    rmse_per_node : liste RMSE par node
    rmse_global   : RMSE global test
    df_preds      : DataFrame ['ID', 'yield_pred']
    """

    n_years, n_nodes, timesteps, n_features = X_test_tensor.shape
    neighbors_idx = get_neighbors_idx(A, n_neighbors)

    rmse_per_node = []
    df_list = []

    # Boucle sur tous les nodes
    for node_id in range(n_nodes):

        model = models[node_id]
        if model is None:
            raise ValueError(f"⚠️ Le modèle du node {node_id} est None")

        # voisins
        selected_ids = [node_id] + neighbors_idx[node_id]

        # Extraire la fenêtre locale
        Xn = tf.gather(X_test_tensor, selected_ids, axis=1)
        Xn = tf.reshape(Xn, (n_years, timesteps, (n_neighbors + 1) * n_features))

        # vrai rendement
        yn = y_test_tensor[:, node_id, :]

        # prédiction
        y_pred = model.predict(Xn, verbose=0)

        # RMSE local
        rmse_node = rmse_tf(yn, y_pred)
        rmse_per_node.append(rmse_node)

        # --- Récupération de l'ID correct depuis id_test ---
        # id_test shape = (years, nodes, 1)
        ids_node = id_test[:, node_id, 0].astype(int)  # longueur = n_years

        df_node = pd.DataFrame({
            "ID": ids_node,
            "yield_pred": y_pred.flatten()
        })

        df_list.append(df_node)

    # DataFrame final prediction
    # y_pred_flat = np.concatenate([df["yield_pred"].values for df in df_list]).astype(np.float32)
    df_preds = pd.concat(df_list, axis=0).reset_index(drop=True)

    # RMSE global
    rmse_global = rmse_df(y_test, df_preds)

    return rmse_per_node, rmse_global, df_preds

# ------------------------
# Prediction
# ------------------------

def predict_local_models(models, X_test_tensor, id_test, A, n_neighbors=5):
    """
    Prédit le rendement pour chaque node à partir de X_test_tensor et retourne un DataFrame.

    Paramètres
    ----------
    models : liste de modèles locaux
    X_test_tensor : np.array ou tf.Tensor, shape = (years, nodes, timesteps, features)
    id_test : np.array shape = (years, nodes, 1) → contient l'ID réel ('Unnamed: 0')
    A : np.array, matrice d'adjacence
    n_neighbors : int, nombre de voisins à utiliser pour chaque node

    Retour
    ------
    df_preds : pd.DataFrame avec colonnes ['ID', 'yield_pred']
    """
    n_years, n_nodes, timesteps, n_features = X_test_tensor.shape
    neighbors_idx = get_neighbors_idx(A, n_neighbors)

    df_list = []

    for node_id in range(n_nodes):
        model = models[node_id]
        if model is None:
            continue  # on saute les nodes sans modèle

        # récupérer les voisins
        selected_ids = [node_id] + neighbors_idx[node_id]

        # extraire les features locales
        Xn = tf.gather(X_test_tensor, selected_ids, axis=1)
        Xn = tf.reshape(Xn, (n_years, timesteps, (n_neighbors + 1) * n_features))

        # prédiction
        y_pred = model.predict(Xn, verbose=0)

        # récupérer les ID pour ce node
        ids_node = id_test[:, node_id, 0].astype(int)

        # créer un DataFrame pour ce node
        df_node = pd.DataFrame({
            "ID": ids_node,
            "yield_pred": y_pred.flatten()
        })

        df_list.append(df_node)

    # concaténer tous les nodes
    df_preds = pd.concat(df_list, axis=0).reset_index(drop=True)

    return df_preds


# ------------------------
# Pipeline
# ------------------------

def pipeline_nodes(X_train, y_train, X_val, y_val, X_test, y_test,
                   n_neighbors=5, nb_features=7, batch_nodes=16, batch_size=2,
                   epochs=20) :

    print("\n Lancement du pipeline_nodes...")

    # --- Fix problème d'arrondi ---
    X_train['lat_orig'] = X_train['lat_orig'].round(6)
    X_train['lon_orig'] = X_train['lon_orig'].round(6)

    X_val['lat_orig'] = X_val['lat_orig'].round(6)
    X_val['lon_orig'] = X_val['lon_orig'].round(6)

    X_test['lat_orig'] = X_test['lat_orig'].round(6)
    X_test['lon_orig'] = X_test['lon_orig'].round(6)

    # --- Création de A prenant en compte l'ensemble des points géographiques ---
    print("\n Création de la matrice A...")
    X_full = pd.concat([X_train, X_val, X_test], axis=0)
    coord_all, A_all = matrice_adj(X_full, n_neighbors)

    # --- Préprocessing ---
    print("\n Lancement du preprocessing des set train, val et test ...")
    print("\n Train...")
    X_tensor_train, y_tensor_train, id_train = preproc_nodes(X_train, y_train, coord_all, A_all, nb_features, test=False)

    print("\n Val...")
    X_tensor_val, y_tensor_val, id_val = preproc_nodes(X_val, y_val, coord_all, A_all, nb_features, test=False)

    print("\n Test...")
    X_tensor_test, y_tensor_test, id_test = preproc_nodes(X_test, y_test, coord_all, A_all, nb_features, test=True)

    # --- Models par node ---
    models = train_local_models_batched(
        X_tensor_train, y_tensor_train,
        X_tensor_val, y_tensor_val,
        A_all, n_neighbors,
        batch_nodes,
        epochs,
        batch_size)

    # --- Evaluation ---
    print("\n Lancement de l'évaluation sur le test...")
    rmse_per_node, rmse_global, preds_all = evaluate_local_models(
         models,
         X_tensor_test,
         y_tensor_test,
         y_test,
         id_test,
         A_all,
         n_neighbors)

    # --- Output ---

    print("\n🎉 Models fit et evalués !")

    print(f"\n RMSE Global sur l'ensemble des nodes : {round(rmse_global, 3)}")

    return models, coord_all, rmse_per_node, rmse_global, preds_all


def pipeline_nodes_all(crop, X_train, y_train, X_test,
                   n_neighbors=5, nb_features=7, batch_nodes=16, batch_size=2,
                   epochs=20) :

    print("\n Lancement du pipeline_nodes...")

    # --- Fix problème d'arrondi ---
    X_train['lat_orig'] = X_train['lat_orig'].round(6)
    X_train['lon_orig'] = X_train['lon_orig'].round(6)

    # --- Création de A prenant en compte l'ensemble des points géographiques ---
    print("\n Création de la matrice A...")
    coord_all, A_all = matrice_adj(X_train, n_neighbors)

    # --- Préprocessing ---
    print("\n Lancement du preprocessing du train")
    X_tensor_train, y_tensor_train, id_train = preproc_nodes(X_train, y_train, coord_all, A_all, nb_features, test=False)


    # --- Models par node ---
    models = train_local_models_batched_all(
        X_tensor_train, y_tensor_train,
        A_all, n_neighbors,
        batch_nodes,
        epochs,
        batch_size)

    print("\n🎉 Models fit et evalués !")

    # --- Predict ---
    print("\n Lancement predict")
    X_test_tensor, id_test = preproc_nodes_x(X_test, coord_all, A_all, nb_features, test=False)
    y_pred = predict_local_models(models, X_test_tensor, id_test, A_all, n_neighbors)
    y_pred.set_index('ID', inplace=True)

    # --- Enregistrer en CSV ---
    print("\n Enregistrement csv")
    X_test_set = X_test[["ID", "real_year", "lon_orig","lat_orig"]]
    y_pred_df = X_test_set.merge(y_pred, how='left', left_on='ID', right_index=True)

    y_pred_df.to_csv(f"y_pred_{crop}_new.csv")

    return y_pred_df


import gc

def pipeline_nodes_all_low_memory(
    crop,
    X_train,
    y_train,
    X_test,
    n_neighbors=5,
    nb_features=7,
    batch_nodes=16,
    batch_size=2,
    epochs=20
):
    print("\n Lancement du pipeline_nodes...")

    # --- Fix problème d'arrondi ---
    X_train['lat_orig'] = X_train['lat_orig'].round(6)
    X_train['lon_orig'] = X_train['lon_orig'].round(6)

    # --- Création de A prenant en compte l'ensemble des points géographiques ---
    print("\n Création de la matrice A...")
    coord_all, A_all = matrice_adj(X_train, n_neighbors)

    # --- Préprocessing TRAIN ---
    print("\n Lancement du preprocessing du train")
    X_tensor_train, y_tensor_train, id_train = preproc_nodes(
        X_train, y_train, coord_all, A_all, nb_features, test=False
    )

    # 🔻 On peut déjà libérer les DataFrames train (ils ne servent plus après le préproc)
    del X_train, y_train

    # --- Models par node ---
    models = train_local_models_batched_all(
        X_tensor_train, y_tensor_train,
        A_all, n_neighbors,
        batch_nodes,
        epochs,
        batch_size
    )

    # 🔻 Libérer les tensors de train (on ne garde que les modèles)
    del X_tensor_train, y_tensor_train, id_train
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    print("\n🎉 Models fit et evalués !")

    # --- Préparer le subset de X_test AVANT de le transformer en tenseurs ---
    X_test_set = X_test[["ID", "real_year", "lon_orig", "lat_orig"]].copy()

    # --- Préprocessing TEST ---
    print("\n Lancement du preprocessing du test")
    X_test_tensor, id_test = preproc_nodes_x(
        X_test, coord_all, A_all, nb_features, test=False
    )

    # 🔻 On peut libérer le gros DataFrame X_test maintenant
    del X_test

    # --- Predict ---
    print("\n Lancement predict")
    y_pred = predict_local_models(models, X_test_tensor, id_test, A_all, n_neighbors)
    y_pred.set_index('ID', inplace=True)

    # 🔻 Libération des tensors test + matrices de graphes
    print("\n Réduction de la mémoire")
    del X_test_tensor, id_test, A_all, coord_all
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # --- Fusion finale ---
    print(f"\n Création de y_pred_{crop}")
    y_pred_df = X_test_set.merge(y_pred, how='left', left_on='ID', right_index=True)

    # 🔻 On a merge, donc X_test_set et y_pred ne sont plus nécessaires
    print("\n Réduction de la mémoire")
    del X_test_set, y_pred

    # --- Enregistrer en CSV en local (sur la VM) ---
    print("\n Enregistrement csv")
    out_path = f"y_pred_{crop}_full.csv"
    print(f"\n💾 Sauvegarde de {out_path} de taille {y_pred_df.shape} ...")

    # chunksize pour écrire par blocs si le DF est énorme
    y_pred_df.to_csv(out_path, index=False, chunksize=50_000)
    print("\n✅ CSV sauvegardé.\n")

    # Si tu n'as PAS besoin du DataFrame en RAM sur la VM,
    # tu peux retourner seulement le chemin du fichier :
    # return out_path

    return y_pred_df


