import os
import pandas as pd
import gc
import gcsfs  # à ajouter si pas déjà fait

from future_crop.ml_logic.model_local import pipeline_nodes_all

import warnings
warnings.filterwarnings("ignore")

def run():

    CROP = "maize"

    BUCKET_NAME = os.getenv("BUCKET_NAME", "future-crop-bucket")
    BASE = f"gs://{BUCKET_NAME}/processed_data"

    X_train_path = f"{BASE}/X_train_{CROP}_full.csv"
    y_train_path = f"{BASE}/y_train_{CROP}_full.csv"
    X_test_path  = f"{BASE}/X_test_{CROP}_full.csv"

    print("🔹 Loading data...")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    X_test  = pd.read_csv(X_test_path)
    

    print("Shapes :", X_train.shape, y_train.shape, X_test.shape)

    y_pred = pipeline_nodes_all(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        n_neighbors=5,
        nb_features=7,
        batch_nodes=32,
        batch_size=4,
        epochs=15,
    )
    

if __name__ == "__main__":
    run()
    
    
    