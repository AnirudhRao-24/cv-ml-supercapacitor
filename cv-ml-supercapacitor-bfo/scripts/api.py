from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Import TensorFlow to load the .keras model
from tensorflow.keras.models import load_model

# Initialize the API
app = FastAPI()

# --- Explicit CORS Whitelist ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://anirudhrao-24.github.io",
        "https://anirudhrao-24.github.io/",
        "http://localhost:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
    ], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dynamic Absolute Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '../models')

print("Loading ML Pipeline into memory...")
scaler = joblib.load(os.path.join(MODELS_DIR, 'standard_scaler.pkl'))

# Load all 3 Base Models
rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl'))
ann_model = load_model(os.path.join(MODELS_DIR, 'ann_model.keras'))

# Load the Meta-Model
meta_model = joblib.load(os.path.join(MODELS_DIR, 'meta_stacked_model.pkl'))

class CVRequest(BaseModel):
    features: list

@app.get("/")
def home():
    return {
        "status": "Online",
        "project": "BiFeO3 ML-Supercapacitor Pipeline",
        "message": "API is active. Send POST requests to /predict to execute the model."
    }

@app.post("/predict")
def predict_current(request: CVRequest):
    # 1. Scale the raw chemistry data (6 features)
    input_data = np.array(request.features)
    scaled_data = scaler.transform(input_data)
    
    # 2. LEVEL 0: Generate predictions from the Base Models
    # .flatten() is used because neural networks output 2D arrays, but sklearn needs 1D
    ann_pred = ann_model.predict(scaled_data).flatten()
    rf_pred = rf_model.predict(scaled_data)
    xgb_pred = xgb_model.predict(scaled_data)
    
    # 3. LEVEL 1: Stack the 3 predictions into a new feature matrix
    meta_features = np.column_stack((ann_pred, rf_pred, xgb_pred))
    
    # 4. LEVEL 2: The Meta-Model makes the final prediction
    final_prediction = meta_model.predict(meta_features)
    
    # Extract Feature Importance for the UI
    feature_weights = rf_model.feature_importances_.tolist()
    
    return {
        "predicted_current": final_prediction.tolist(),
        "feature_importance": feature_weights
    }