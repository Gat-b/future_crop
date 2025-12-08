import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from keras import Model, Sequential
from keras.layers import Input, Lambda, TimeDistributed, Masking, ConvLSTM2D, GlobalAveragePooling1D, Conv1D, Flatten, Dense, MaxPooling3D, Dropout, Bidirectional, LayerNormalization, MultiHeadAttention
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy import sparse
from spektral.layers import GCNConv, GATConv
import tensorflow as tf
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from sklearn.neighbors import BallTree
from future_crop.ml_logic.gru_lstm import *

### Préproc train et val ###

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
        for d, j in zip(distances_km[i][1:], indices[i][1:]):  # ignorer soi-même
            # Poids possible : 1 / distance
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
    return tf.convert_to_tensor(X_filled, dtype=tensor.dtype)

def preproc_gcn(X, y, coord, A, nb_features=7):
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
            X_series = np.array(x_node[:-4]).reshape(nb_features, 240)

            # Injection dans le tenseur final
            X_tensor[id_year, node_id] = X_series

        for y_node in y_year.itertuples(index=False):# Boucle sur les noeuds géographiques pour y
            # Trouver le noeud correspondant dans les nodes uniques
            #print(y_node)
            lat = getattr(y_node, 'lat_orig')
            lon = getattr(y_node, 'lon_orig')
            node_id = coord_index[(lat, lon)]

            # Extraction du rendement
            # last_col_name = y_node.columns[-1]
            # y_value = y_node[last_col_name]
            y_value = getattr(y_node, '_4') #Nom de la colonne 'yield' TODO comprendre pourquoi ça s'est changé

            # Injection dans le tenseur final
            y_tensor[id_year, node_id] = y_value

    ## Ajout des valeurs manquantes par les moyennes des voisins
    X_tensor = impute_neighbors(X_tensor, A)
    y_tensor = impute_neighbors(y_tensor, A)

    #X_tensor = tf.transpose(X_tensor, perm=(0,1,3,2))  # → (years, nodes, 240, features)

    print("Shape de X_tensor:", X_tensor.shape)
    print("Shape de y_tensor:", y_tensor.shape)

    return X_tensor, y_tensor

def pipeline_preproc_gcn(X_train, y_train, X_val, y_val, coord_all, A_all, nb_features=7):
    """
    Prépare les tenseurs X, y et les matrices d'adjacence A pour l'entraînement et la validation
    de façon robuste, sans dépendre des arguments keyword.
    """

    # # --- Création de la matrice d'adjacence complète ---
    # X_full = pd.concat([X_train, X_val_full], axis=0)
    # coord_all, A_all = matrice_adj(X_full, n_neighbors)

    # --- Prétraitement train ---
    X_tensor_train, y_tensor_train = preproc_gcn(X_train, y_train, coord_all, A_all, nb_features)

    # --- Prétraitement validation ---
    X_tensor_val, y_tensor_val = preproc_gcn(X_val, y_val, coord_all, A_all, nb_features)

    # # --- Batching des matrices A pour train et val ---
    # A_train_batched = np.tile(A_all, (X_tensor_train.shape[0], 1, 1))
    # A_val_batched = np.tile(A_all, (X_tensor_val.shape[0], 1, 1))

    # A_train_batched_tf = tf.constant(A_train_batched, dtype=tf.float32)
    # A_val_batched_tf = tf.constant(A_val_batched, dtype=tf.float32)

    # --- Conversion en tenseurs TensorFlow ---
    # X_tensor_train = tf.convert_to_tensor(X_tensor_train, dtype=tf.float32)
    # y_tensor_train = tf.convert_to_tensor(y_tensor_train, dtype=tf.float32)
    # X_tensor_val = tf.convert_to_tensor(X_tensor_val, dtype=tf.float32)
    # y_tensor_val = tf.convert_to_tensor(y_tensor_val, dtype=tf.float32)

    return X_tensor_train, y_tensor_train, X_tensor_val, y_tensor_val


#### Modèle & Params ###

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=None))

def conv1d_gcn_model(X_tensor):
    # Les dimensions statiques sont toujours déduites de X_tensor.shape
    _, nodes, timesteps, features = X_tensor.shape

    # --- Définition des entrées ---
    X_in = Input(shape=(nodes, timesteps, features))
    A_in = Input((nodes, nodes), sparse=True)

    # Conv1D
    # x = tf.reshape(X_in, (-1, timesteps, features))

    x = TimeDistributed(Conv1D(256, 9, activation="relu", padding="same"))(X_in)
    x = TimeDistributed(Dropout(0.3))(x)

    x = TimeDistributed(Conv1D(128, 7, activation="relu", padding="same"))(x)
    x = TimeDistributed(Dropout(0.3))(x)

    X_nodes = TimeDistributed(GlobalAveragePooling1D())(x)
    # x = Dense(128, activation='relu')(x) # x.shape = (Total_Elements, 64) -> où Total_Elements = batch * nodes

    # --- Reshape pour le GCN (Passage de (Total_Elements, C) à (N, V, C)) ---
    # X_nodes = Lambda(
    #     lambda t: tf.reshape(t, (-1, nodes, 128)),
    #     output_shape=(nodes, 128)
    # )(x)

    # GCN
    H = GCNConv(128)([X_nodes, A_in])
    H = Dropout(0.2)(H)
    H = GCNConv(64)([H, A_in])
    H = Dropout(0.1)(H)

    # Out
    out = Dense(1, activation='linear')(H)

    model = Model(inputs=[X_in, A_in], outputs=out)
    return model

def lstm_model(X_tensor):
    # Les dimensions statiques sont toujours déduites de X_tensor.shape
    _, nodes, timesteps, features = X_tensor.shape

    # --- Définition des entrées ---
    X_in = Input(shape=(nodes, timesteps, features))
    A_in = Input((nodes, nodes), sparse=True)

    # --- LSTM ---

    x = TimeDistributed(LSTM(256))(X_in)
    x = TimeDistributed(Dropout(0.3))(x)

    # --- Out ---
    out = Dense(1, activation='linear')(x)

    model = Model(inputs=[X_in, A_in], outputs=out)

    return model

def lstm_gcn_model(X_tensor):
    # Les dimensions statiques sont toujours déduites de X_tensor.shape
    _, nodes, timesteps, features = X_tensor.shape

    # --- Définition des entrées ---
    X_in = Input(shape=(nodes, timesteps, features))
    A_in = Input((nodes, nodes), sparse=True)

    # --- LSTM ---

    x = TimeDistributed(LSTM(256))(X_in)
    x = TimeDistributed(Dropout(0.3))(x)

    # --- GCN ---
    H = GCNConv(128)([x, A_in])
    H = Dropout(0.2)(H)
    H = GCNConv(64)([H, A_in])
    H = Dropout(0.1)(H)

    # --- Out ---
    out = Dense(1, activation='linear')(H)

    model = Model(inputs=[X_in, A_in], outputs=out)

    return model


def simple_gcn_lstm_model(X_tensor, A):
    """
    GCN → LSTM simple sans Spektral
    X_tensor: (batch, nodes, timesteps, features)
    A: (nodes, nodes) adjacency matrix dense pré-normalisée
    """
    batch, nodes, timesteps, features = X_tensor.shape

    X_in = Input(shape=(nodes, timesteps, features))

    # Transpose pour (batch, timesteps, nodes, features)
    x = tf.transpose(X_in, perm=(0, 2, 1, 3))

    # --- GCN simple par timestep ---
    # H = A X W
    def gcn_layer(H, units, dropout=0.0):
        H = tf.linalg.matmul(A, H)              # propagation
        H = Dense(units, activation='relu')(H)  # transformation
        if dropout > 0:
            H = Dropout(dropout)(H)
        return H

    gcn_outputs = []
    for t in range(timesteps):
        H = gcn_layer(x[:, t], 128, 0.2)
        H = gcn_layer(H, 64, 0.1)
        gcn_outputs.append(H)

    # Stack → (batch, timesteps, nodes, features_gcn)
    x = tf.stack(gcn_outputs, axis=1)

    # Transpose → (batch, nodes, timesteps, features_gcn)
    x = tf.transpose(x, perm=(0, 2, 1, 3))

    # --- LSTM temporel par node ---
    x = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False)))(x)
    x = TimeDistributed(Dropout(0.3))(x)

    # --- Sortie ---
    out = Dense(1, activation='linear')(x)

    model = Model(inputs=X_in, outputs=out)
    return model


def bidir_lstm_gcn_model(X_tensor):
    # Dimensions statiques
    _, nodes, timesteps, features = X_tensor.shape

    # --- Définition des entrées ---
    X_in = Input(shape=(nodes, timesteps, features))  # (batch, nodes, timesteps, features)
    A_in = Input((nodes, nodes), sparse=True)

    # --- Bidirectional LSTM par node ---
    # LSTM renvoie un embedding par node
    x = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False)))(X_in)  # (batch, nodes, 256)
    x = TimeDistributed(Dropout(0.3))(x)

    # --- GCN ---
    H = GCNConv(256)([x, A_in])
    H = Dropout(0.2)(H)
    H = GCNConv(128)([H, A_in])
    H = Dropout(0.1)(H)

    # --- Output par node ---
    out = Dense(1, activation='linear')(H)  # (batch, nodes, 1)

    model = Model(inputs=[X_in, A_in], outputs=out)
    return model

def bidir_lstm_gat_model(X_tensor, n_heads=4):
    # Dimensions statiques
    _, nodes, timesteps, features = X_tensor.shape

    # --- Entrées ---
    X_in = Input(shape=(nodes, timesteps, features))  # (batch, nodes, timesteps, features)
    A_in = Input((nodes, nodes), sparse=True)        # adjacency matrix

    # --- Bidirectional LSTM par node ---
    x = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False)))(X_in)  # (batch, nodes, 256)
    x = TimeDistributed(Dropout(0.3))(x)

    X_nodes = x  # embeddings temporels par node

    # --- GAT ---
    H = GATConv(128, attn_heads=n_heads, concat_heads=True)([X_nodes, A_in])
    H = Dropout(0.2)(H)

    H = GATConv(64, attn_heads=n_heads, concat_heads=True)([H, A_in])
    H = Dropout(0.1)(H)

    # --- Sortie par node ---
    out = Dense(1, activation='linear')(H)  # (batch, nodes, 1)

    model = Model(inputs=[X_in, A_in], outputs=out)
    return model

def conv1d_lstm_gcn_model(X_tensor):
    """
    Modèle Conv1D → LSTM → GCN léger, adapté à (batch, nodes, timesteps, features)
    """
    _, nodes, timesteps, features = X_tensor.shape

    # --- Entrées ---
    X_in = Input(shape=(nodes, timesteps, features))  # (batch, nodes, timesteps, features)
    A_in = Input((nodes, nodes), sparse=True)        # adjacency matrix

    # --- Conv1D temporelle (motifs locaux) ---
    x = TimeDistributed(Conv1D(256, kernel_size=5, padding='same', activation='relu'))(X_in)
    x = TimeDistributed(Dropout(0.3))(x)

    # --- LSTM temporelle (dépendances long-terme) ---
    x = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True)))(x)  # réduit timesteps à embedding
    x = TimeDistributed(Dropout(0.3))(x)


    # # --- Réduction temporelle (évite explosions mémoire) ---
    # x_mean = tf.reduce_mean(x, axis=2)   # (batch, nodes, features)

    # # --- GCN (propagation spatiale) ---
    # H = GCNConv(128)([x_mean, A_in])
    # H = Dropout(0.2)(H)
    # H = GCNConv(128)([H, A_in])
    # H = Dropout(0.1)(H)

    # --- Sortie par node ---
    out = Dense(1, activation='linear')(x)  # (batch, nodes, 1)

    model = Model(inputs=[X_in, A_in], outputs=out)
    return model


def gcn_conv1d_bilstm_model(X_tensor):
    """
    Modèle GCN -> Conv1D -> BiLSTM
    X_tensor: (years, nodes, features, timesteps)
    """
    years, nodes, features, timesteps = X_tensor.shape

    # --- Inputs ---
    X_in = Input(shape=(nodes, features, timesteps))      # (years, nodes, features, timesteps)
    A_in = Input((nodes, nodes), sparse=True)             # adjacency matrix

    # --- Transposer pour (years, nodes, timesteps, features) ---
    X_trans = tf.transpose(X_in, perm=(0, 1, 3, 2))      # (years, nodes, timesteps, features)

    # --- GCN par timestep (boucle Python) ---
    H_list = []
    for t in range(timesteps):
        H_t = GCNConv(32)([X_trans[:, :, t, :], A_in])
        H_t = Dropout(0.2)(H_t)
        H_t = GCNConv(64)([H_t, A_in])
        H_t = Dropout(0.1)(H_t)
        H_list.append(H_t)

    # Stack → (years, nodes, timesteps, features_gcn)
    H = tf.stack(H_list, axis=2)

    # --- Conv1D temporelle ---
    x = TimeDistributed(Conv1D(32, kernel_size=3, padding='same', activation='relu'))(H)
    x = TimeDistributed(Dropout(0.3))(x)

    # --- BiLSTM temporelle ---
    x = TimeDistributed(Bidirectional(LSTM(32, return_sequences=False)))(x)
    x = TimeDistributed(Dropout(0.3))(x)

    # --- Sortie par node ---
    out = Dense(1, activation='linear')(x)  # (years, nodes, 1)

    model = Model(inputs=[X_in, A_in], outputs=out)
    return model


def conv1d_lstm_attention_model(X_tensor, att_heads=4, att_key_dim=16):
    """
    Modèle Conv1D → LSTM → Attention, adapté à (batch, nodes, timesteps, features)
    """
    _, nodes, timesteps, features = X_tensor.shape

    # --- Entrées ---
    X_in = Input(shape=(nodes, timesteps, features), name="X_in")  # (batch, nodes, timesteps, features)

    # --- Conv1D temporelle (motifs locaux) ---
    x = TimeDistributed(Conv1D(256, kernel_size=5, padding='same', activation='relu'))(X_in)
    x = TimeDistributed(Dropout(0.3))(x)

    # --- LSTM temporelle (dépendances long-terme) ---
    x = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True)))(x)
    x = TimeDistributed(Dropout(0.3))(x)

    # --- Attention multi-head par node (sur la dimension temporelle) ---
    # Réordonne pour que MultiHeadAttention traite timesteps comme sequence
    x_reshaped = tf.reshape(x, (-1, timesteps, 256))  # (batch*nodes, timesteps, features)
    att = MultiHeadAttention(num_heads=att_heads, key_dim=att_key_dim)(x_reshaped, x_reshaped)
    att = LayerNormalization()(att + x_reshaped)
    x_att = tf.reshape(att, (-1, nodes, timesteps, 256))  # (batch, nodes, timesteps, features)

    # --- Sortie par node et timestep ---
    out = TimeDistributed(Dense(1, activation='linear'))(x_att)  # (batch, nodes, timesteps, 1)

    model = Model(inputs=X_in, outputs=out)

    return model


### Pipeline ###

def pipeline_model(model, X_train, y_train, X_val, y_val, A_train, A_val):
    # --- Compile ---
    model.compile(loss='mse',
                optimizer='adam',
                metrics=[rmse])

    # --- Callbacks ---
    es = EarlyStopping(patience=10,
                    restore_best_weights=True,
                    monitor='val_rmse')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=1e-6)

    # --- Fit ---
    result = model.fit(
        x=[X_train, A_train], # X_tensor (15, 7771, 240, 5) et A_batched_tf (15, 7771, 7771)
        y=y_train,
        validation_data=([X_val, A_val], y_val),
        epochs=50,
        batch_size=4, # <- batch_size = 15 pour traiter tout en un seul grand lot
        verbose=1,
        shuffle=False,
        callbacks=[reduce_lr, es]
    )
    return model, result

def pipeline_model_wA(model, X_train, y_train, X_val, y_val):
    # --- Compile ---
    model.compile(loss='mse',
                optimizer='adam',
                metrics=[rmse])

    # --- Callbacks ---
    es = EarlyStopping(patience=10,
                    restore_best_weights=True,
                    monitor='val_rmse')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=1e-6)

    # --- Fit ---
    result = model.fit(
        x=X_train, # X_tensor (15, 7771, 240, 5) et A_batched_tf (15, 7771, 7771)
        y=y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=4, # <- batch_size = 15 pour traiter tout en un seul grand lot
        verbose=1,
        shuffle=False,
        callbacks=[reduce_lr, es]
    )
    return model, result

def finetune_progressive(model, X_train, y_train, A_train, X_val, y_val, A_val,
                         base_lr=1e-4,
                         steps=2,         # nombre d'étapes de dégel progressif
                         epochs_per_step=5):

    # 1) On récupère les couches Conv1D dans l'ordre
    conv_layers = [layer for layer in model.layers if isinstance(layer, Conv1D)]

    # 2) Geler toutes les couches Conv1D au départ
    for layer in conv_layers:
        layer.trainable = False

    # Compiler avec LR bas
    model.compile(
        optimizer=Adam(base_lr),
        loss='mse',
        metrics=[rmse]
    )

    print(f"[INFO] Étape 0 : Conv1D gelées, entraînement du GCN uniquement.")

    # 3) Boucle de dégel progressif
    for step in range(steps + 1):

        # Dé-geler progressivement
        if step > 0 and step <= len(conv_layers):
            conv_layers[-step].trainable = True
            print(f"[INFO] Dé-gel de la couche : {conv_layers[-step].name}")

        # Recompiler le modèle après modification du trainable
        model.compile(
            optimizer=Adam(base_lr * (0.5 ** step)),  # LR décroissante
            loss='mse',
            metrics=[rmse]
        )

        print(f"[INFO] Finetuning - Étape {step}/{steps} (LR={base_lr * (0.5 ** step):.1e})")

        model.fit(
            x=[X_train, A_train],
            y=y_train,
            epochs=epochs_per_step,
            batch_size=4,
            shuffle=False,
            validation_data=([X_val, A_val], y_val),
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
            verbose=1
        )

    return model
