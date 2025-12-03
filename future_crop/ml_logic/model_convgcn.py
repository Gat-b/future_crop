import numpy as np

from keras import Model
from keras.layers import Input, Lambda, TimeDistributed, ConvLSTM2D, Conv1D, Flatten, Dense, MaxPooling3D
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
    coords = X_lat_lon[['lat', 'lon']].to_numpy()
    coords_rad = np.radians(coords) ## Transformation en randians

    knn = NearestNeighbors(n_neighbors=5, metric='haversine')
    knn.fit(coords_rad)
    distances, indices = knn.kneighbors(coords_rad) # Récupération des distances et indices du KNN

    # Création des arrêtes
    edges = []
    for i in range(len(coords)):
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


def conv1d_gcn_model(X_lat_lon, X_time):
    '''
    Transformation des données temporelles de chaque point en embedding vectoriel
    '''

    assert X_time.shape[1:] == (240,8)

    N = X_time.shape[0]

    # Création de la matrice adjacente
    edges = heversine(X_lat_lon)
    A = build_adjacency(edges, N)

    # CONV1D model : Embedding de X_time
    X_in = Input(X_time.shape[1:])  # (240, 8)
    x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(X_in)
    X_time_emb = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)


    # x = Flatten()(x)
    # X_time_emb = Dense(64, activation='relu')(x)

    #GCN model
    # Analogie ligne par ligne :
    # 1. chaque village regarde ses voisins et ajuste sa “connaissance climatique” en combinant la sienne avec celle des villages proches
    # 2. c’est comme un deuxième “tour de discussion” avec les voisins, maintenant chacun sait mieux ce qui se passe autour.
    # 3. Prédiction des rendements
    H = GCNConv(64, activation='relu')([X_time_emb, A])
    H = GCNConv(32, activation='relu')([H, A])
    out = Dense(1, activation='relu')(H)

    model = Model(inputs=X_in, outputs=out)
    return model
