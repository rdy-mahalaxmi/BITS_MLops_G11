#!/usr/bin/env python3
"""
Fix model loading issues by recreating models with proper imports
"""

import sys
import os
import mlflow
sys.path.append('src')

from model_training import FeatureEngineer, ModelTrainer
import joblib
import pandas as pd
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'


def fix_models():
    """Recreate models to fix pickle loading issues"""
    # print("Fixing model loading issues...")
    
    # Create MLflow experiment if it doesn't exist
    try:
        mlflow.set_experiment("heart_disease_prediction")
    except:
        mlflow.create_experiment("heart_disease_prediction")
        mlflow.set_experiment("heart_disease_prediction")
    
    # Load the data
    data = pd.read_csv('data/heart_disease.csv')
    
    # Create and train feature engineer
    fe = FeatureEngineer()
    X, y = fe.fit_transform(data)
    
    # Create and train model
    trainer = ModelTrainer()
    results = trainer.train_models(X, y)
    
    # Save models (trainer.best_model is set in train_models method)
    joblib.dump(trainer.best_model, 'models/best_model.pkl')
    joblib.dump(fe, 'models/feature_engineer.pkl')
    
    # print("Models fixed and saved successfully!")
    return True

if __name__ == "__main__":
    fix_models()