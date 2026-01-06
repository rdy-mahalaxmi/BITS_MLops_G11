from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import logging
import uvicorn
from typing import Dict, Any
import os
import time
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the FeatureEngineer class for pickle loading
# Make sure the classes are available in the global namespace for pickle
import sys
sys.path.append(os.path.dirname(__file__))

try:
    from model_training import FeatureEngineer, ModelTrainer
    # Make classes available globally for pickle
    globals()['FeatureEngineer'] = FeatureEngineer
    globals()['ModelTrainer'] = ModelTrainer
except ImportError as e:
    logger.error(f"Failed to import model classes: {e}")
    FeatureEngineer = None
    ModelTrainer = None

# Simple metrics storage
metrics = defaultdict(int)
start_time = time.time()

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

# Load models at startup
try:
    # Try loading from current directory first, then from parent directory
    model_paths = [
        'models/best_model.pkl',
        '../models/best_model.pkl',
        os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
    ]
    
    fe_paths = [
        'models/feature_engineer.pkl',
        '../models/feature_engineer.pkl', 
        os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_engineer.pkl')
    ]
    
    model = None
    feature_engineer = None
    
    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            break
    
    for path in fe_paths:
        if os.path.exists(path):
            feature_engineer = joblib.load(path)
            break
    
    if model and feature_engineer:
        logger.info("Models loaded successfully")
    else:
        logger.warning("Could not load all models")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    model = None
    feature_engineer = None

class PredictionInput(BaseModel):
    age: float
    sex: int
    chest: int
    resting_blood_pressure: float
    serum_cholestoral: float
    fasting_blood_sugar: int
    resting_electrocardiographic_results: int
    maximum_heart_rate_achieved: float
    exercise_induced_angina: int
    oldpeak: float
    slope: int
    number_of_major_vessels: float
    thal: float

class PredictionOutput(BaseModel):
    prediction: int
    confidence: float
    risk_level: str

@app.get("/")
async def root():
    return {"message": "Heart Disease Prediction API", "status": "healthy"}

@app.get("/health")
async def health_check():
    if model is None or feature_engineer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput) -> PredictionOutput:
    metrics["requests"] += 1
    start_time = time.time()
    try:
        if model is None or feature_engineer is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Log request with timestamp
        request_log = {
            "timestamp": datetime.now().isoformat(),
            "request_data": input_data.dict(),
            "client_info": "api_request"
        }
        logger.info(f"Prediction request: {request_log}")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Add dummy target for feature engineering
        input_df['target'] = 0
        
        # Transform features
        X = feature_engineer.transform(input_df)
        
        # Make prediction
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X)[0].max()
        
        # Determine risk level
        if confidence > 0.8:
            risk_level = "High Confidence"
        elif confidence > 0.6:
            risk_level = "Medium Confidence"
        else:
            risk_level = "Low Confidence"
        
        result = PredictionOutput(
            prediction=int(prediction),
            confidence=float(confidence),
            risk_level=risk_level
        )
        
        # Log response with metrics
        processing_time = time.time() - start_time
        response_log = {
            "timestamp": datetime.now().isoformat(),
            "prediction": int(prediction),
            "confidence": float(confidence),
            "processing_time_ms": round(processing_time * 1000, 2),
            "status": "success"
        }
        logger.info(f"Prediction result: {response_log}")
        metrics["success"] += 1
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_log = {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "processing_time_ms": round(processing_time * 1000, 2),
            "status": "error"
        }
        logger.error(f"Prediction error: {error_log}")
        metrics["errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Enhanced metrics endpoint for monitoring"""
    uptime = time.time() - start_time
    return {
        "uptime_seconds": uptime,
        "total_requests": metrics["requests"],
        "successful_predictions": metrics["success"],
        "failed_predictions": metrics["errors"],
        "model_loaded": model is not None,
        "feature_engineer_loaded": feature_engineer is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    