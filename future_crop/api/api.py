
### Import the necessary libraries ###

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from google.cloud import storage

### Loading package functions and dfs ###
# from future_crop.ml_logic.function import *
from future_crop.params import RESULT_PATH_STORAGE, BUCKET_NAME

app = FastAPI(title="Future Crop API")

### Adding middleware ###

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### API ###
@app.get("/")
def root():
    return {
        "status": "API is running",
        "message": "Use /yield?model=knn to get data"
    }

@app.get("/yield")
def get_yield(model: str = "knn", crop:str = "wheat"):
    """
    Returns the pre-calculated dataframe for the selected model.
    """
    filename = f"{crop}_{model}_yield_pred.csv.gz"
    blob_path = f"yield_forecasts/{filename}"  # Path inside the bucket

    try:
        # 1. Initialize GCS Client
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_path)

        # 2. Download the compressed file as bytes
        # Since the file is small (2-5MB), we can download into memory safely
        file_content = blob.download_as_bytes()

        # 3. Return the raw bytes with the correct headers
        # This sends the small compressed file, bypassing the 32MB limit
        return Response(
            content=file_content,
            media_type="application/gzip",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        return {
            "error": "File not found or unreadable",
            "details": str(e)
        }

### Prediction ###



# @app.get("/predict")
# def predict(
#     model_name: str = app.model, #rf
#     mean_yield_loc: float, #5.4
#     lon: float, #2.15
#     soil_co2_nitrogen: float, #18.7
#     mean_pr: float, #1.2
#     sum_pr: float, #320
#     lat: float, #48.85
#     sum_rsds: float, #36000
#     mean_rsds: float, #165
#     soil_co2_co2: float, #410
#     min_rsds: float, #12
#     ):
#     """
#     Predict yield for a single location using a chosen model.

#     Parameters
#     ----------
#     model_name : str
#         Model identifier, expected values: "lgbm" or "rf".
#     mean_yield_loc, lon, soil_co2_nitrogen, mean_pr, sum_pr, lat,
#     sum_rsds, mean_rsds, soil_co2_co2, min_rsds : float
#         Numeric feature values required for prediction.

#     Returns
#     -------
#     dict
#         JSON-serializable response with keys:
#         - "model": chosen model name
#         - "yield": predicted yield (float)
#     """
    


#     data = {"mean_yield_loc": [mean_yield_loc],
#             "lon": [lon],
#             "soil_co2_nitrogen": [soil_co2_nitrogen],
#             "mean_pr": [mean_pr],
#             "sum_pr": [sum_pr],
#             "lat": [lat],
#             "sum_rsds": [sum_rsds],
#             "mean_rsds": [mean_rsds],
#             "soil_co2_co2": [soil_co2_co2],
#             "min_rsds": [min_rsds],
#             }

#     df = pd.DataFrame(data)

#     for col in features_name_ml_reviewed:
#         if col not in df.columns:
#             df[col] = 0.0

#     X = df[features_name_ml_reviewed]
#     y_pred = model_predict(X, model)

#     return {
#         "model": model_name,
#         "yield": float(y_pred[0])
#     }

