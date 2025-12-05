import numpy as np

from keras import Model
from keras.layers import Input, Lambda, TimeDistributed, Masking, ConvLSTM2D, Conv1D, Flatten, Dense, MaxPooling3D
from keras.callbacks import EarlyStopping


from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy import sparse
from spektral.layers import GCNConv
import tensorflow as tf

'''
    Legende :
    - X_time : shape ( _, 240, 8 )
    - X_lat_lon : tableau avec les latitude et longitude
    - A matrice d'adjacency
    - X_time_emb : sortie emb de X_time après conv1D
'''


def heversine(X_lat_lon):
    '''
    Calcul de la disctance sur une sphère à partir des lat et long de X
    X[['lat', 'long']]
    '''

    # Création pour chaque point des indices des 5 points les plus proches et
    # de leurs distances en km
    coords_rad = np.radians(X_lat_lon) ## Transformation en randians

    knn = NearestNeighbors(n_neighbors=5, metric='haversine')
    knn.fit(coords_rad)
    distances, indices = knn.kneighbors(coords_rad) # Récupération des distances et indices du KNN

    # Création des arrêtes
    edges = []
    for i in range(len(X_lat_lon)):
        for j in indices[i]:
            edges.append((i, j))

    return edges


def build_adjacency(edges, N):
    """
    edges : liste de tuples (i,j)
    N : nombre de points

    return : matrice A (adjacency) pour le model GNN
    """

    # Construire la matrice scipy sparse d'abord
    adjacency = sparse.lil_matrix((N, N))
    for i, j in edges :
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    # Convertir en coo_matrix pour récupérer row/col/data
    adjacency_coo = adjacency.tocoo()

    # Conversion en SparseTensor TensorFlow
    adjacency_tf = tf.sparse.SparseTensor(
        indices=np.vstack((adjacency_coo.row, adjacency_coo.col)).T,
        values=adjacency_coo.data.astype(np.float32),
        dense_shape=adjacency_coo.shape
    )
    adjacency_tf = tf.sparse.reorder(adjacency_tf)

    return adjacency_tf


def padding(X_time, y):
    '''
    Padding des années avec des NaN pour avoir un autant de point de données
    par an et donc nue shape constante.
    '''

    N_max = max([X.shape[0] for X in X_time]) # Le nombre max de points (stations) sur l'ensemble des années


    # Padding pour X
    X_reshaped_time_padded = []
    y_reshaped_padded = []

    for i, year_data in enumerate(X_time):
            N_current = year_data.shape[0]  # Le nombre actuel de stations pour cette année

            # Padding pour X_time
            if N_current < N_max:
                # Compléter avec des NaNs
                padding = np.full((N_max - N_current, 240, 5), np.nan)  # Compléter avec NaN
                # Ajouter le padding à l'année
                padded_year_data = np.vstack([year_data, padding])  # Ajouter les stations manquantes
            else:
                padded_year_data = year_data  # Si N_current == N_max, pas besoin de padding

            X_reshaped_time_padded.append(padded_year_data)

            # Padding pour y (si y est fourni)
            if y is not None:
                y_current = y[i]  # Récupérer les valeurs cibles pour l'année en cours
                if y_current.shape[0] < N_max:
                    # Compléter y avec NaNs (ou une autre valeur comme 0)
                    y_padding = np.full((N_max - y_current.shape[0], y_current.shape[1]), np.nan)
                    padded_y = np.vstack([y_current, y_padding])  # Ajouter le padding
                else:
                    padded_y = y_current  # Si y_current == N_max, pas de padding nécessaire

                y_reshaped_padded.append(padded_y)

    # Convertir X_reshaped_time_padded en numpy array
    X_reshaped_time_padded = np.array(X_reshaped_time_padded)
    y_reshaped_padded = np.array(y_reshaped_padded)

    return X_reshaped_time_padded, y_reshaped_padded
