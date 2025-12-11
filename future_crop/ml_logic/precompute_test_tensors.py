import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf

from future_crop.ml_logic.model_local import ( 
    matrice_adj,
    preproc_nodes,
)

BUCKET_NAME = os.getenv("BUCKET_NAME", "future-crop-bucket")
BASE = f"gs://{BUCKET_NAME}/processed_data"

def precompute_for_crop(crop: str, n_neighbors: int = 5, nb_features: int = 7):
    print(f"\n=== Pré-calcul complet pour {crop.upper()} ===")

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

    # --- Arrondis pour cohérence ---
    for df in [X_train, X_test]:
        df["lat_orig"] = df["lat_orig"].round(6)
        df["lon_orig"] = df["lon_orig"].round(6)

    # --- Matrice d'adjacence ---
    print("\n Création de la matrice A...")
    coord_all, A_all = matrice_adj(X_train, n_neighbors=n_neighbors)

    # --- PREPROCESS TRAIN ---
    print("\n Lancement du preprocessing TRAIN...")
    X_train_tensor, y_train_tensor, _ = preproc_nodes(
        X_bef=X_train,
        y_bef=y_train,
        coord=coord_all,
        A=A_all,
        nb_features=nb_features,
        test=False
    )

    print("Shapes TRAIN :")
    print("  X_train_tensor:", X_train_tensor.shape)
    print("  y_train_tensor:", y_train_tensor.shape)

    # --- PREPROCESS TEST ---
    print("\n Lancement du preprocessing TEST...")
    X_test_tensor, y_dummy, _ = preproc_nodes(
        X_bef=X_test,
        y_bef=y_train,     # y_dummy inutile
        coord=coord_all,
        A=A_all,
        nb_features=nb_features,
        test=True
    )

    print("Shapes TEST :")
    print("  X_test_tensor:", X_test_tensor.shape)

    # --- Convert tensors to numpy ---
    def to_np(x):
        return x.numpy() if isinstance(x, tf.Tensor) else x

    X_train_np = to_np(X_train_tensor)
    y_train_np = to_np(y_train_tensor)
    X_test_np  = to_np(X_test_tensor)
    A_np       = np.array(A_all)
    coord_np   = coord_all.to_numpy()

    # --- SAVE ---
    out_dir = "./precomputed_tensors"
    os.makedirs(out_dir, exist_ok=True)

    print("\n💾 Sauvegarde des tenseurs...")

    np.save(os.path.join(out_dir, f"X_train_{crop}.npy"), X_train_np)
    np.save(os.path.join(out_dir, f"y_train_{crop}.npy"), y_train_np)
    np.save(os.path.join(out_dir, f"X_test_{crop}.npy"),  X_test_np)

    # Matrice A + coords
    np.save(os.path.join(out_dir, f"A_all_{crop}.npy"), A_np)
    np.save(os.path.join(out_dir, f"coord_all_{crop}.npy"), coord_np)

    print(f"✅ Sauvegardé pour {crop} dans {out_dir}/")

    # --- Clean memory ---
    del (X_train, y_train, X_test, coord_all, A_all,
         X_train_tensor, y_train_tensor, X_test_tensor,
         X_train_np, y_train_np, X_test_np, A_np, coord_np)
    gc.collect()


def main():
    for crop in ["wheat", "maize"]:
        precompute_for_crop(crop)


if __name__ == "__main__":
    main()
