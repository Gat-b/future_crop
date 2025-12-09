import pandas as pd
from pathlib import Path
import gc


def build_train_val_all_crops(crops=["wheat", "maize"]):

    project_root = Path(__file__).resolve().parents[2]
    base = project_root / "processed_data"

    all_dfs = []

    for crop in crops:
        # Charge X et y
        X_train = pd.read_csv(base / f"X_train_{crop}_explo.csv")
        X_val   = pd.read_csv(base / f"X_val_{crop}_explo.csv")
        y_train = pd.read_csv(base / f"y_train_{crop}_explo.csv")
        y_val   = pd.read_csv(base / f"y_val_{crop}_explo.csv")

        # jointure sur ID pour récupérer la colonne yield
        train = X_train.merge(y_train, on="ID")
        val   = X_val.merge(y_val, on="ID")

        # concat train + val pour cette culture
        df_crop = pd.concat([train, val], axis=0)
        all_dfs.append(df_crop)

    # concat blé + maïs
    full_df = pd.concat(all_dfs, axis=0)

    # Sauvegarde
    output_path = base / "train_val_all_crops.csv"
    full_df.to_csv(output_path, index=False)

    print(f"Shape globale : {full_df.shape}")
    print(f"Fichier créé : {output_path}")

    return full_df


def build_train_full_all_crops():

    root = Path(__file__).resolve().parents[2]
    base_path = root/"processed_data"

    all_dfs = []

    for crop in ["wheat", "maize"]:
        print(f"Traitement train_full pour {crop}...")
        X_train = pd.read_csv(base_path/f"X_train_{crop}_full.csv")
        y_train = pd.read_csv(base_path/f"y_train_{crop}_full.csv")

        full_crop_df = X_train.merge(y_train, on="ID")
        all_dfs.append(full_crop_df)

    full_train_df = pd.concat(all_dfs, axis=0)

    output_path = base_path/"train_full_all_crops.csv"
    full_train_df.to_csv(output_path, index=False)

    return full_train_df


def build_test_all_crops():

    root = Path(__file__).resolve().parents[2]
    base_path = root/"processed_data"

    all_dfs = []

    for crop in ["wheat", "maize"]:
        X_test = pd.read_csv(base_path/f"X_test_{crop}_full.csv")
        all_dfs.append(X_test)

    full_test_df = pd.concat(all_dfs, axis=0)

    output_path = base_path/"X_test_all_crops_full.csv"
    full_test_df.to_csv(output_path, index=False)

    return full_test_df

if __name__ == "__main__":
    # build_train_val_all_crops()
    # build_train_full_all_crops()
    build_test_all_crops()





# def build_train_val_all_crops():
#     root = Path(__file__).resolve().parents[2]
#     base_path = root/"processed_data"
#     output_path = base_path/"train_val_all_crops.csv"

#     first = True
#     #all_df = []

#     for crop in ["wheat", "maize"]:
#         print(f"Traitement train/val pour {crop}...")
#         X_train = pd.read_csv(base_path/f"X_train_{crop}_explo.csv")
#         X_val = pd.read_csv(base_path/f"X_val_{crop}_explo.csv")
#         y_train = pd.read_csv(base_path/f"y_train_{crop}_explo.csv")
#         y_val = pd.read_csv(base_path/f"y_val_{crop}_explo.csv")

#         train = X_train.merge(y_train, on="ID")
#         val = X_val.merge(y_val, on="ID")

#         df_crop = pd.concat([train, val], axis=0)

#         df_crop.to_csv(
#             output_path,
#             mode="w" if first else "a",
#             header=first,
#             index=False
#         )

#         first = False

#         del X_train, X_val, y_train, y_val, train, val, df_crop
#         gc.collect()

#     print(f"Fichier train+val multi-crops créé : {output_path}")

# def build_test_all_crops():

#     root = Path(__file__).resolve().parents[2]
#     base_path = root/"processed_data"
#     output_path = base_path/"X_test_all_crops_full.csv"

#     first = True

#     for crop in ["wheat", "maize"]:
#         print(f"Traitement test pour {crop}...")
#         X_test = pd.read_csv(base_path/f"X_test_{crop}_full.csv")


#         X_test.to_csv(
#             output_path,
#             mode="w" if first else "a",
#             header=first,
#             index=False
#         )

#         first = False

#         del X_test
#         gc.collect()

#     print(f"Fichier test multi-crops créé : {output_path}")
