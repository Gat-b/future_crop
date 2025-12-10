import tensorflow as tf
from keras.layers import Input, ConvLSTM2D, Dropout, Dense, Lambda
from keras.models import Model
from spektral.layers import GCNConv
from keras import backend as K
from future_crop.ml_logic.preproc_model_gnc import matrice_adj, preproc_gcn
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np

# --- Métrique RMSE compatible ---
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=None))

# --- ConvLSTM + GCN ---
def convlstm_gcn_model(X_tensor, nodes=None):
    """
    X_tensor: (batch, nodes, timesteps, features)
    """
    batch, N_nodes, T, F = X_tensor.shape
    nodes = N_nodes if nodes is None else nodes

    # Inputs
    X_in = Input(shape=(nodes, T, F))            # (batch, nodes, timesteps, features)
    A_in = Input((nodes, nodes), sparse=False)   # matrice d'adjacence

    # Reshape pour ConvLSTM2D : (batch*nodes, timesteps, 1, 1, features)
    x = Lambda(lambda t: K.reshape(t, (-1, T, 1, 1, F)))(X_in)

    # ConvLSTM2D
    x = ConvLSTM2D(filters=128, kernel_size=(1,1), activation='tanh', return_sequences=False)(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)

    # Reshape pour GCN : (batch, nodes, features)
    X_nodes = Lambda(lambda t: K.reshape(t, (-1, nodes, 128)))(x)

    # GCN
    H = GCNConv(128, activation='relu')([X_nodes, A_in])
    H = Dropout(0.2)(H)
    H = GCNConv(64, activation='relu')([H, A_in])
    H = Dropout(0.1)(H)

    # Sortie
    out = Dense(1, activation='linear')(H)

    model = Model(inputs=[X_in, A_in], outputs=out)
    return model

# --- Préprocessing avec batchage correct ---
def pipeline_preproc_gcn(X_train, y_train, X_val, y_val, nb_features=7, n_neighbors=5):
    # Concat pour coordonnées uniques
    X_tot = pd.concat([X_train, X_val], axis=0)
    coord, A_dense_2d = matrice_adj(X_tot, n_neighbors)
    A_dense_2d = A_dense_2d.astype(np.float32)

    # Préproc train et val
    X_tensor_train, y_tensor_train = preproc_gcn(X_train, y_train, coord, A_dense_2d, nb_features)
    X_tensor_val, y_tensor_val = preproc_gcn(X_val, y_val, coord, A_dense_2d, nb_features)

    # Batchage de A
    A_train_batched = np.tile(A_dense_2d, (X_tensor_train.shape[0], 1, 1))
    A_val_batched   = np.tile(A_dense_2d, (X_tensor_val.shape[0], 1, 1))

    A_train_batched_tf = tf.constant(A_train_batched, dtype=tf.float32)
    A_val_batched_tf   = tf.constant(A_val_batched, dtype=tf.float32)

    # Conversion en tensors
    X_tensor_train = tf.convert_to_tensor(X_tensor_train)
    y_tensor_train = tf.convert_to_tensor(y_tensor_train)
    X_tensor_val   = tf.convert_to_tensor(X_tensor_val)
    y_tensor_val   = tf.convert_to_tensor(y_tensor_val)

    return X_tensor_train, y_tensor_train, X_tensor_val, y_tensor_val, A_train_batched_tf, A_val_batched_tf

# --- Pipeline pour compilation et fit ---
def pipeline_model_convlstm(X_train, y_train, X_val, y_val, nb_features=7, n_neighbors=5, patience=10):
    # Préprocessing
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, A_train, A_val = \
        pipeline_preproc_gcn(X_train, y_train, X_val, y_val, nb_features, n_neighbors)

    # Modèle
    model = convlstm_gcn_model(X_train_tensor)

    # Compilation
    model.compile(loss='mse', optimizer='adam', metrics=[rmse])

    # Early stopping
    es = EarlyStopping(patience=patience, restore_best_weights=True, monitor='val_rmse')

    # Fit
    result = model.fit(
        x=[X_train_tensor, A_train],
        y=y_train_tensor,
        validation_data=([X_val_tensor, A_val], y_val_tensor),
        epochs=50,
        batch_size=1,
        callbacks=[es],
        verbose=1
    )

    return model, result
