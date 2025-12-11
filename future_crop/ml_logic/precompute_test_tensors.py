import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf

from future_crop.ml_logic.model_local import (  # ← adapte le chemin si besoin
    matrice_adj,
    preproc_nodes,
)

BUCKET_NAME = os.getenv("BUCKET_NAME", "future-crop-bucket")
BASE = f"gs://{BUCKET_NAME}/processed_data"

def precompute_X_test_for_crop(crop: str, n_neighbors: int = 5, nb_features: int = 7):
    print(f"\n=== Pré-calcul de X_test pour {crop.upper()} ===")

    # --- Paths ---
    X_train_path = f"{BASE}/X_train_{crop}_full.csv"
    y_train_path = f"{BASE}/y_train_{crop}_full.csv"
    X_test_path  = f"{BASE}/X_test_{crop}_full.csv"

    # --- Load data ---
    print(f"\n 🔹 Loading data : X_train_{crop}...")
    X_train = pd.read_csv(X_train_path)

    print(f"\n 🔹 Loading data : y_train_{crop}...")
    y_train = pd.read_csv(y_train_path)

    print(f"\n 🔹 Loading data : X_test_{crop}...")
    X_test  = pd.read_csv(X_test_path)

    print("Shapes :", X_train.shape, y_train.shape, X_test.shape)

    # --- Fix arrondis pour cohérence avec le train ---
    X_train["lat_orig"] = X_train["lat_orig"].round(6)
    X_train["lon_orig"] = X_train["lon_orig"].round(6)
    X_test["lat_orig"]  = X_test["lat_orig"].round(6)
    X_test["lon_orig"]  = X_test["lon_orig"].round(6)

    # --- Matrice d'adjacence commune (basée sur le train) ---
    print("\n Création de la matrice A...")
    coord_all, A_all = matrice_adj(X_train, n_neighbors=n_neighbors)

    # --- Préprocess complet (X + y + id) en mode test=True ---
    # Ici on ne se sert que de X_tensor et id, mais preproc_nodes a besoin de y.
    print("\n Lancement du preprocessing du TEST (full)...")
    X_test_tensor, y_dummy, id_test = preproc_nodes(
        X_bef=X_test,
        y_bef=y_train,        # on recycle y_train, y_dummy ne sera pas utilisé
        coord=coord_all,
        A=A_all,
        nb_features=nb_features,
        test=True,
    )

    print("✅ Preprocessing test terminé.")
    print("Shape X_test_tensor :", X_test_tensor.shape)
    print("Shape id_test       :", id_test.shape)

    # --- Sauvegarde sur la VM en .npy ---
    out_dir = f"./precomputed_tensors"
    os.makedirs(out_dir, exist_ok=True)

    # X_test_tensor est un tf.Tensor → on le convertit en numpy
    X_test_np = X_test_tensor.numpy() if isinstance(X_test_tensor, tf.Tensor) else X_test_tensor

    X_out_path  = os.path.join(out_dir, f"X_test_{crop}_tensor.npy")
    id_out_path = os.path.join(out_dir, f"id_test_{crop}.npy")

    print(f"\n💾 Sauvegarde de {X_out_path} ...")
    np.save(X_out_path, X_test_np)

    print(f"💾 Sauvegarde de {id_out_path} ...")
    np.save(id_out_path, id_test)

    # Libération mémoire
    del X_train, y_train, X_test, coord_all, A_all, X_test_tensor, y_dummy, id_test, X_test_np
    gc.collect()

    print(f"\n✅ Pré-calcul terminé pour {crop.upper()}.\n")


def main():
    for crop in ["wheat", "maize"]:
        precompute_X_test_for_crop(crop)


if __name__ == "__main__":
    main()
