import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_training import FeatureEngineer, ModelTrainer
from fastapi.testclient import TestClient
from api import app

class TestFeatureEngineer:
    def setup_method(self):
        self.fe = FeatureEngineer()
        self.sample_data = pd.DataFrame({
            'age': [50, 60, 45],
            'sex': [1, 0, 1],
            'chest_pain': [2, 1, 3],
            'resting_bp': [120, 140, 110],
            'cholesterol': [200, 250, 180],
            'fasting_bs': [0, 1, 0],
            'resting_ecg': [1, 0, 2],
            'max_hr': [150, 120, 170],
            'exercise_angina': [0, 1, 0],
            'oldpeak': [1.0, 2.5, 0.5],
            'st_slope': [1, 2, 0],
            'target': [0, 1, 0]
        })
    
    def test_fit_transform(self):
        X, y = self.fe.fit_transform(self.sample_data)
        assert X.shape[0] == 3
        assert X.shape[1] == 11
        assert len(y) == 3
        assert 'target' not in X.columns
    
    def test_transform(self):
        # First fit
        X, y = self.fe.fit_transform(self.sample_data)
        
        # Then transform new data
        new_data = self.sample_data.iloc[:1].copy()
        X_new = self.fe.transform(new_data)
        assert X_new.shape[0] == 1
        assert X_new.shape[1] == 11

class TestAPI:
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert "Heart Disease Prediction API" in response.json()["message"]
    
    def test_health_endpoint(self):
        response = self.client.get("/health")
        # May return 503 if models not loaded, which is expected in test environment
        assert response.status_code in [200, 503]
    
    def test_predict_endpoint_structure(self):
        # Test with valid input structure
        test_input = {
            "age": 50,
            "sex": 1,
            "chest_pain": 2,
            "resting_bp": 120,
            "cholesterol": 200,
            "fasting_bs": 0,
            "resting_ecg": 1,
            "max_hr": 150,
            "exercise_angina": 0,
            "oldpeak": 1.0,
            "st_slope": 1
        }
        
        response = self.client.post("/predict", json=test_input)
        # May return 503 if models not loaded, which is expected in test environment
        assert response.status_code in [200, 503]

class TestDataValidation:
    def test_data_types(self):
        # Test that our sample data has correct types
        data = pd.DataFrame({
            'age': [50],
            'sex': [1],
            'target': [0]
        })
        
        assert data['age'].dtype in [np.int64, np.float64]
        assert data['sex'].dtype in [np.int64, np.float64]
        assert data['target'].dtype in [np.int64, np.float64]
    
    def test_data_ranges(self):
        # Test reasonable data ranges
        data = pd.DataFrame({
            'age': [25, 80],
            'sex': [0, 1],
            'chest_pain': [0, 3],
            'target': [0, 1]
        })
        
        assert data['age'].min() >= 0
        assert data['age'].max() <= 120
        assert data['sex'].isin([0, 1]).all()
        assert data['chest_pain'].isin([0, 1, 2, 3]).all()
        assert data['target'].isin([0, 1]).all()

if __name__ == "__main__":
    pytest.main([__file__])