import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from keras import Model, Sequential
from keras.layers import Input, Lambda, TimeDistributed, Masking, ConvLSTM2D, GlobalAveragePooling1D, Conv1D, Flatten, Dense, MaxPooling3D, Dropout
from keras.callbacks import EarlyStopping


from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy import sparse
from spektral.layers import GCNConv
import tensorflow as tf
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from sklearn.neighbors import BallTree


def matrice_adj(X, n_neighbors=5):
    """
    Retourne :
        - coord : DataFrame unique (lat, lon)
        - A : matrice d'adjacence pondérée selon distance haversine
    """

    # Extraire coordonnées uniques
    coord = X[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
    print(f"Nombre de noeuds uniques : {len(coord)}")

    # Conversion en radians pour BallTree
    coord_rad = np.radians(coord[['lat', 'lon']].values)

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
        for d, j in zip(distances_km[i][1:], indices[i][1:]):  # ignorer soi-même
            # Poids possible : 1 / distance
            A[i, j] = 1 / (d + 1e-6)

    return coord, A




def preproc_gcn(X, y, coord, A, nb_features=5):
    '''
    Docstring pour preproc_gcn_X

    :param X: X_train or X_val
    :param y: y_train or y_val
    :param features_num: 5
    :param coord: pd.df coord (lat, lon) unique
    '''
    ### Params
    years = sorted(X['real_year'].unique())
    nb_years = len(years)
    nb_nodes = len(coord)

    #### X PREPROCESS
    ### Set index
    X_set = X.set_index('real_year')

    ### X_time : sélection des colonnes à garder
    # Ici on garde 'lat' et 'long' pour pouvoir faire l'ajout de celles manquantes
    selected_cols = list(['lat', 'lon']) + list(X_set.columns[252:])
    X_time = X_set[selected_cols]
    X_time.drop(columns=['soil_co2_co2', 'soil_co2_nitrogen'], inplace=True)

    #### Y PREPROCESS
    # Objectif : relier les données y avec la lat et la lon
    X_coord = X[['ID', 'lat', 'lon', 'real_year']]
    y_set = X_coord.merge(y)
    y_set.drop(columns=['ID'], inplace=True)


    ### Ajout des lat et long manquantes
    # 1. Initalisation avec des 0
    X_tensor = np.full((nb_years, nb_nodes, nb_features, 240), 0, dtype=float)
    y_tensor = np.full((nb_years, nb_nodes, 1), 0, dtype=float)

    # 2. Création d'un dico pour ajouter l'indice de la coordonnées > Permet davoir toujours le même ordre pour la matrice A
    coord_index = {(lat, lon): i for i, (lat, lon) in enumerate(coord[['lat', 'lon']].itertuples(index=False))}

    # 3. Création du tenseur final
    for id_year, year in enumerate(years) : # Boucle sur les années

        X_year = X_time[X_time.index == year]
        y_year = y_set[y_set['real_year'] == year]

        for x_node in X_year.itertuples(index=False):# Boucle sur les noeuds géographiques pour X
            # Trouver le noeud correspondant dans les nodes uniques
            lat = getattr(x_node, 'lat')
            lon = getattr(x_node, 'lon')
            node_id = coord_index[(lat, lon)]

            # Extraction de la série temporelle (240 jours, features_num)
            X_series = np.array(x_node[2:]).reshape(nb_features, 240)
            y_series = np.array(x_node[2:]).reshape(nb_features, 240)

            # Injection dans le tenseur final
            X_tensor[id_year, node_id] = X_series

        for y_node in y_year.itertuples(index=False):# Boucle sur les noeuds géographiques pour y
            # Trouver le noeud correspondant dans les nodes uniques
            lat = getattr(y_node, 'lat')
            lon = getattr(y_node, 'lon')
            node_id = coord_index[(lat, lon)]

            # Extraction du rendement
            y_value = getattr(y_node, '_3') #Nom de la colonne 'yield' TODO comprendre pourquoi ça s'est changé

            # Injection dans le tenseur final
            y_tensor[id_year, node_id] = y_value



    ## Reshape de X_reshaped_time des features en (240, features_num)
    # X_reshaped_time = X_reshaped_time.reshape(15, 7603, 240, features_num)

    print("Shape de X_tensor:", X_tensor.shape)
    print("Shape de y_tensor:", y_tensor.shape)

    ## Ajout des valeurs manquantes par les moyennes des voisins
    X_tensor = impute_neighbors(X_tensor, A)
    y_tensor = impute_neighbors(y_tensor, A)

    return X_tensor, y_tensor


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
    return tf.convert_to_tensor(X_filled, dtype=tensor.dtype)


def pipeline_preproc_gcn(X_train, y_train, X_val, y_val, nb_features=5, n_neighbors=5) :

    # Création de la matrice A sur l'ensemble des X pour avoir toutes les coordonnées possibles
    X_tot = pd.concat([X_train, X_val], axis=0)
    coord, A = matrice_adj(X_tot, n_neighbors)

    # Création des X et y tensor avec un padding sur la moyenne des 5 voisins
    X_tensor_train, y_tensor_train = preproc_gcn(X_train, y_train, coord, A, nb_features)
    X_tensor_val, y_tensor_val = preproc_gcn(X_val, y_val, coord, A, nb_features)

    # Conversion en TensorFow
    X_tensor_train = tf.convert_to_tensor(X_tensor_train)
    y_tensor_train = tf.convert_to_tensor(y_tensor_train)
    X_tensor_val = tf.convert_to_tensor(X_tensor_val)
    y_tensor_val = tf.convert_to_tensor(y_tensor_val)

    ### Conversion de A en tf.constant
    N_batch = X_tensor_train.shape[0]
    A_dense_2d = A.astype(np.float32)  # Forme: (7771, 7771)
    A_dense_batched = np.tile(A_dense_2d, (N_batch, 1, 1)) # Forme: (15, 7771, 7771)
    A_dense_batched_tf = tf.constant(A_dense_batched, dtype=tf.float32)

    return X_tensor_train, y_tensor_train, X_tensor_val, y_tensor_val, A_dense_batched_tf


def conv1d_gcn_model(X_tensor):
    # Les dimensions statiques sont toujours déduites de X_tensor.shape
    _, nodes, timesteps, features = X_tensor.shape

    # --- Définition des entrées ---
    X_in = Input(shape=(nodes, timesteps, features))
    A_in = Input((nodes, nodes), sparse=True)

    # Conv1D
    x = tf.reshape(X_in, (-1, timesteps, features))
    x = Conv1D(256, 9, activation='relu', padding='same')(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, 7, activation='relu', padding='same')(x)
    x = Dropout(0.3)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x) # x.shape = (Total_Elements, 64) -> où Total_Elements = batch * nodes

    # --- Reshape pour le GCN (Passage de (Total_Elements, C) à (N, V, C)) ---
    X_nodes = Lambda(
        lambda t: K.reshape(t, (K.shape(t)[0] // nodes, nodes, 128)),
        # On peut aussi utiliser le tuple (-1, nodes, 64) si Keras le supporte dans ce contexte
        # lambda t: tf.reshape(t, (-1, nodes, 64))
        output_shape=(nodes, 128)
    )(x)

    # GCN
    H = GCNConv(128)([X_nodes, A_in])
    H = Dropout(0.2)(H)
    H = GCNConv(64)([H, A_in])
    H = Dropout(0.1)(H)

    # Out
    out = Dense(1, activation='linear')(H)

    model = Model(inputs=[X_in, A_in], outputs=out)
    return model

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def pipeline_model(X_train, y_train, X_val, y_val, A, patience=10):
    es = EarlyStopping(
    patience=patience,
    restore_best_weights=True,
    monitor='val_rmse'
    )

    lr_scheduler = LearningRateScheduler(lr_schedule)

    #Initialisation du modele
    model = conv1d_gcn_model(X_train)

    # Compilation du modèle
    model.compile(loss='mse',
                optimizer='adam',
                metrics=[rmse])

    # FIt du modèle
    result = model.fit(
        x=[X_train, A], # X_tensor (15, 7771, 240, 5) et A_batched_tf (15, 7771, 7771)
        y=y_train,
        validation_data=([X_val, A], y_val),
        epochs=50,
        batch_size=1, # <- batch_size = 15 pour traiter tout en un seul grand lot
        verbose=1,
        callbacks=[lr_scheduler, es]
    )
    return result

def lr_schedule(epoch):
    '''
    Docstring pour lr_schedule

    Décroissance learning rate en fonction du nombre d'epoch
    '''
    if epoch < 15:
        return 0.001
    elif epoch < 30:
        return 0.0005
    else:
        return 0.0001
