
### Import the necessary libraries ###

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

### Loading package functions and dfs ###
from future_crop.ml_logic.function import *
from future_crop.params import *

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

@app.get("/yield")
def get_yield(model: str = "lgbm"):
    """
    Returns the pre-calculated dataframe for the selected model.
    """
    # Construct the path dynamically based on the model parameter
    filename = f"{model}_yield_pred.csv"
    path = os.path.join(RESULT_PATH_STORAGE, filename)
    
    if os.path.exists(path): 
        result_df = pd.read_csv(path)
        # CONVERSION REQUIRED: FastAPI needs a dict/list, not a DataFrame
        return result_df.to_dict(orient='records') 
    else:
        return {"error": f"File not found for model: {model} at path: {path}"}


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

