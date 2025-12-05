import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from keras import Model, Sequential
from keras.layers import Input, Lambda, TimeDistributed, Masking, ConvLSTM2D, Conv1D, Flatten, Dense, MaxPooling3D
from keras.callbacks import EarlyStopping


from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy import sparse
from spektral.layers import GCNConv
import tensorflow as tf
from keras import backend as K

from future_crop.ml_logic.mask_layers import *



def matrice_adj(X, n_neighbors=5):
    '''
    Preproc X

    Return pd.df coord (lat, lon) unique & adjacent matrix
    '''

    # Tuple de coordonnées uniques
    coord = X[['lat', 'lon']]
    coord.drop_duplicates(inplace=True)
    print(f'Nombre de noeuds uniques : {len(coord)}')

    # Crétion de la matrice adjacente pour le modèle GCN
    A = kneighbors_graph(coord, n_neighbors, metric='haversine')

    return coord, A


def preproc_gcn(X, y, coord, nb_features=5):
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
    # 1. Initalisation avec des NaN
    X_tensor = np.full((nb_years, nb_nodes, 240, nb_features), -1, dtype=float)
    y_tensor = np.full((nb_years, nb_nodes, 1), -1, dtype=float)

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
            X_series = np.array(x_node[2:]).reshape(240, nb_features)
            y_series = np.array(x_node[2:]).reshape(240, nb_features)

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

    return X_tensor, y_tensor


def pipeline_preproc_gcn(X_train, y_train, X_val, y_val, nb_features=5) :

    # Création de la matrice A sur l'ensemble des X pour avoir toutes les coordonnées possibles
    X_tot = pd.concat([X_train, X_val], axis=0)
    coord, A = matrice_adj(X_tot, n_neighbors=5)

    # Création des X et y tensor
    X_tensor_train, y_tensor_train = preproc_gcn(X_train, y_train, coord, nb_features)
    X_tensor_val, y_tensor_val = preproc_gcn(X_val, y_val, coord, nb_features)

    # conversion en TensorFow
    X_tensor_train = tf.convert_to_tensor(X_tensor_train)
    y_tensor_train = tf.convert_to_tensor(y_tensor_train)
    X_tensor_val = tf.convert_to_tensor(X_tensor_val)
    y_tensor_val = tf.convert_to_tensor(y_tensor_val)

    ### Conversion de A en tf.constant
    N_batch = X_tensor_train.shape[0]
    A_dense_2d = A.toarray().astype(np.float32)  # Forme: (7771, 7771)
    A_dense_batched = np.tile(A_dense_2d, (N_batch, 1, 1)) # Forme: (15, 7771, 7771)
    A_dense_batched_tf = tf.constant(A_dense_batched, dtype=tf.float32)

    return X_tensor_train, y_tensor_train, X_tensor_val, y_tensor_val, A_dense_batched_tf


def conv1d_gcn_model(X_tensor):
    # Les dimensions statiques sont toujours déduites de X_tensor.shape
    _, nodes, timesteps, features = X_tensor.shape

    # --- Définition des entrées ---
    X_in = Input(shape=(nodes, timesteps, features))
    A_in = Input((nodes, nodes), sparse=True)

    # --- Conv1D Part ---
    # x = (None * nodes, timesteps, features)
    x = tf.reshape(X_in, (-1, timesteps, features))
    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x) # x.shape = (Total_Elements, 64) -> où Total_Elements = batch * nodes

    # --- Reshape pour le GCN (Passage de (Total_Elements, C) à (N, V, C)) ---

    # K.shape(t)[0] est la taille totale (batch * nodes)
    # K.shape(t)[0] // nodes est la taille du batch dynamique

    X_nodes = Lambda(
        lambda t: K.reshape(t, (K.shape(t)[0] // nodes, nodes, 64)),
        # On peut aussi utiliser le tuple (-1, nodes, 64) si Keras le supporte dans ce contexte
        # lambda t: tf.reshape(t, (-1, nodes, 64))
        output_shape=(nodes, 64)
    )(x)

    # GCN
    H = GCNConv(64)([X_nodes, A_in])
    H = GCNConv(32)([H, A_in])
    out = Dense(1, activation='relu')(H)

    model = Model(inputs=[X_in, A_in], outputs=out)
    return model
