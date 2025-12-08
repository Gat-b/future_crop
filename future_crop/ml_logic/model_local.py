import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv1D, LSTM, Bidirectional, Dropout, Dense, TimeDistributed
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

# ------------------------
# Identification des voisins
# ------------------------
def get_neighbors_idx(A, k=5):
    neighbors_idx = {}
    for i in range(A.shape[0]):
        row = np.array(A[i]).flatten()
        sorted_idx = np.argsort(-row)
        top_k = [int(idx) for idx in sorted_idx if idx != i][:k]
        neighbors_idx[i] = top_k
    return neighbors_idx

# ------------------------
# Modèle Conv1D → LSTM
# ------------------------

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=None))

def conv1d_lstm_model(input_shape):

    X_in = Input(shape=input_shape)

    # x = Conv1D(64, kernel_size=5, padding='same', activation='relu')(X_in)
    # x = Dropout(0.3)(x)

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

def train_local_models(X_tensor_train, y_tensor_train, X_tensor_val, y_tensor_val, A, k=5, epochs=10, batch_size=2):
    n_years, n_nodes, timesteps, n_features = X_tensor_train.shape
    neighbors_idx = get_neighbors_idx(A, k)
    models = []

    for node_id in range(n_nodes):
        print(f"Avancement: {node_id + 1} / {n_nodes}")
        # Node + k voisins
        selected_ids = [node_id] + neighbors_idx[node_id]

        # Extraire node + voisins et fusionner features
        X_train_node = tf.gather(X_tensor_train, indices=selected_ids, axis=1)  # (years, k+1, timesteps, features)
        X_train_node = tf.reshape(X_train_node, (n_years, timesteps, (k+1)*n_features))
        y_train_node = y_tensor_train[:, node_id, :]  # (years, 1)

        X_val_node = tf.gather(X_tensor_val, indices=selected_ids, axis=1)
        X_val_node = tf.reshape(X_val_node, (X_val_node.shape[0], timesteps, (k+1)*n_features))
        y_val_node = y_tensor_val[:, node_id, :]

        # --- Callbacks ---
        es = EarlyStopping(patience=10,
                        restore_best_weights=True,
                        monitor='val_rmse')

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=2, min_lr=1e-6)

        # Construire modèle
        model = conv1d_lstm_model(input_shape=(timesteps, (k+1)*n_features))

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
        A, k=5,
        batch_nodes=16,
        epochs=10,
        batch_size=2):

    n_years, n_nodes, timesteps, n_features = X_tensor_train.shape
    neighbors_idx = get_neighbors_idx(A, k)

    models = [None] * n_nodes

    print(f"\n### Entraînement batché : {batch_nodes} modèles en parallèle ###\n")

    # --- boucle par batch de nodes ---
    for start in tqdm(range(0, n_nodes, batch_nodes), desc="Training"):
        end = min(start + batch_nodes, n_nodes)
        batch_ids = list(range(start, end))

        # --- préparer inputs batchés ---
        X_train_batch = []
        y_train_batch = []
        X_val_batch = []
        y_val_batch = []

        for node_id in batch_ids:
            selected_ids = [node_id] + neighbors_idx[node_id]

            # train
            Xn = tf.gather(X_tensor_train, selected_ids, axis=1)
            Xn = tf.reshape(Xn, (n_years, timesteps, (k+1) * n_features))

            yn = y_tensor_train[:, node_id, :]

            X_train_batch.append(Xn)
            y_train_batch.append(yn)

            # val
            Xn_val = tf.gather(X_tensor_val, selected_ids, axis=1)
            Xn_val = tf.reshape(Xn_val, (Xn_val.shape[0], timesteps, (k+1) * n_features))

            yn_val = y_tensor_val[:, node_id, :]

            X_val_batch.append(Xn_val)
            y_val_batch.append(yn_val)

        # Convertir en tensor batchés
        X_train_batch = tf.stack(X_train_batch)  # shape = (B, years, timesteps, features)
        y_train_batch = tf.stack(y_train_batch)  # shape = (B, years, 1)
        X_val_batch = tf.stack(X_val_batch)
        y_val_batch = tf.stack(y_val_batch)

        # On doit fusionner (B * years) pour entraîner un seul modèle partagé
        B = len(batch_ids)

        X_train_flat = tf.reshape(X_train_batch, (B * n_years, timesteps, (k+1) * n_features))
        y_train_flat = tf.reshape(y_train_batch, (B * n_years, 1))

        X_val_flat = tf.reshape(X_val_batch, (B * X_val_batch.shape[1], timesteps, (k+1) * n_features))
        y_val_flat = tf.reshape(y_val_batch, (B * X_val_batch.shape[1], 1))

        # --- callbacks ---
        es = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_rmse')
        rl = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3)

        # --- modèle partagé pour ce batch ---
        model = conv1d_lstm_model((timesteps, (k+1) * n_features))

        model.fit(
            X_train_flat, y_train_flat,
            validation_data=(X_val_flat, y_val_flat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es, rl],
            verbose=0
        )

        # enregistrer ce modèle pour tous les nodes du batch
        for i, node_id in enumerate(batch_ids):
            models[node_id] = model  # même modèle pour ce batch

    print("\n🎉 Training terminé !")
    return models


# ------------------------
# Evaluation des models
# ------------------------

def evaluate_local_models(
        models,
        X_tensor_full,
        y_tensor_full,
        A,
        k=5
    ):
    """
    Évalue tous les modèles locaux sur X_tensor_full / y_tensor_full
    Retourne :
      - rmse_per_node : liste de rmse pour chaque node
      - rmse_global   : rmse sur tous les nodes concaténés
      - preds_all     : array de toutes les prédictions shape (years, nodes, 1)
    """

    n_years, n_nodes, timesteps, n_features = X_tensor_full.shape

    # voisins
    def get_neighbors_idx(A, k):
        A_np = np.array(A)
        neighbors_idx = {}
        for i in range(n_nodes):
            sorted_idx = np.argsort(-A_np[i])
            neighbors = [j for j in sorted_idx if j != i]
            neighbors_idx[i] = neighbors[:k]
        return neighbors_idx

    neighbors_idx = get_neighbors_idx(A, k)

    preds_all = np.zeros((n_years, n_nodes, 1))
    rmse_per_node = []

    # ----- boucle nodes -----
    for node_id in range(n_nodes):

        model = models[node_id]
        if model is None:
            raise ValueError(f"⚠️ Le modèle du node {node_id} est None")

        selected_ids = [node_id] + neighbors_idx[node_id]

        # Input shape : (years, timesteps, features*(k+1))
        Xn = tf.gather(X_tensor_full, selected_ids, axis=1)
        Xn = tf.reshape(
            Xn, (n_years, timesteps, (k + 1) * n_features)
        )

        yn = y_tensor_full[:, node_id, :]  # (years, 1)

        # prédictions
        y_pred = model.predict(Xn, verbose=0)
        preds_all[:, node_id, :] = y_pred

        rmse_node = rmse(yn, y_pred)
        rmse_per_node.append(rmse_node)

    # RMSE global
    rmse_global = rmse(
            tf.reshape(y_tensor_full, [-1]),
            tf.reshape(preds_all, [-1]))

    return rmse_per_node, rmse_global, preds_all
