from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pathlib import Path
import yaml
import pandas as pd
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ml_bike_demand_end_to_end.pipelines.nodes import (
    load_model, predict, join_timestamps, rename_columns, get_features
)

model = None
params = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, params
    params_path = project_root / "conf" / "base" / "parameters.yml"
    with open(params_path) as f:
        params = yaml.safe_load(f)
        
    model_type = params["training"]["model_type"]
    model_storage = params["model_storage"]
    model_storage["path"] = str(project_root / model_storage["path"])
    
    model = load_model(model_type, model_storage)
    print("Model loaded successfully on startup!")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def make_prediction(request: Request):
    data = await request.json()
    df = pd.DataFrame(data)
    
    # Feature engineering params
    rename_dict = params["feature_engineering"]["rename_columns"]
    lag_params = params["feature_engineering"]["lag_params"]
    
    # Process batch using the Kedro nodes
    df_renamed = rename_columns(df, rename_dict)
    features, timestamps = get_features(df_renamed, lag_params)
    predictions = predict(model, features)
    result = join_timestamps(predictions, timestamps)
    
    # Convert datetime to string for JSON
    result["datetime"] = result["datetime"].astype(str)
    return result.to_dict(orient="records")