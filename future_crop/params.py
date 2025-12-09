import os
import numpy as np

##################  VARIABLES  ################## TBU
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT")

GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
# MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
# MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
# MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
# PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
# PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")

PLATFORM = os.environ.get("PLATFORM")

##################  CONSTANTS  #####################
if PLATFORM == "local":
    LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "gat_b", "future_crop", "raw_data")
    LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "gat_b", "future_crop", "processed_data")
    MODEL_PATH_STORAGE = os.path.join(os.path.expanduser('~'), "code", "gat_b", "future_crop", "models")
    RESULT_PATH_STORAGE = os.path.join(os.path.expanduser('~'), "code", "gat_b", "future_crop", "yield_forecasts")

elif PLATFORM == "docker":
    # In Docker, we will create a folder called 'app' that contains everything
    base_path = "/app"
    LOCAL_DATA_PATH = os.path.join(base_path, "raw_data")
    LOCAL_REGISTRY_PATH = os.path.join(base_path, "processed_data")
    MODEL_PATH_STORAGE = os.path.join(base_path, "models")
    RESULT_PATH_STORAGE = os.path.join(base_path, "yield_forecasts")

elif PLATFORM == "gcp":
    pass



DTYPES_PROCESSED = np.float32


################## VALIDATIONS #################
