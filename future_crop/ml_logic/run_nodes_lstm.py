import os
import pandas as pd
import gc
import gcsfs  
import torch

from future_crop.ml_logic.model_local import pipeline_nodes_all_low_memory

import warnings
warnings.filterwarnings("ignore")

def run():

    BUCKET_NAME = os.getenv("BUCKET_NAME", "future-crop-bucket")
    BASE = f"gs://{BUCKET_NAME}/processed_data"
    
    results = {}
    for crop in ['wheat', 'maize'] :
        print(f"--- {crop} ---")

        X_train_path = f"{BASE}/X_train_{crop}_full.csv"
        y_train_path = f"{BASE}/y_train_{crop}_full.csv"
        X_test_path  = f"{BASE}/X_test_{crop}_full.csv"

        print(f"\n 🔹 Loading data : X_train_{crop}...")
        X_train = pd.read_csv(X_train_path)
        print(f"\n 🔹 Loading data : y_train_{crop}...")
        y_train = pd.read_csv(y_train_path)
        print(f"\n 🔹 Loading data : X_test_{crop}...")
        X_test  = pd.read_csv(X_test_path)

        print("\n Shapes :", X_train.shape, y_train.shape, X_test.shape)

        results[crop] = pipeline_nodes_all_low_memory(
            crop=crop,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            n_neighbors=5,
            nb_features=7,
            batch_nodes=32,
            batch_size=4,
            epochs=15,
        )
    
    return results



    

if __name__ == "__main__":
    run()
    
    