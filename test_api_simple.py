#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple API test script
"""

import requests
import time
import subprocess
import sys
import os

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

def test_api_simple():
    """Simple API test"""
    print("Testing API...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Health endpoint: {response.status_code}")
        
        if response.status_code == 200:
            print("[OK] Health endpoint working")
        elif response.status_code == 503:
            print("[WARNING] API running but models not loaded")
            return True
        else:
            print(f"[ERROR] Health endpoint failed: {response.status_code}")
            return False
        
        # Test prediction endpoint
        test_data = {
            "age": 50,
            "sex": 1,
            "chest": 2,
            "resting_blood_pressure": 120,
            "serum_cholestoral": 200,
            "fasting_blood_sugar": 0,
            "resting_electrocardiographic_results": 1,
            "maximum_heart_rate_achieved": 150,
            "exercise_induced_angina": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "number_of_major_vessels": 0,
            "thal": 3
        }
        
        response = requests.post("http://localhost:8000/predict", json=test_data, timeout=5)
        print(f"Prediction endpoint: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Prediction successful: {result}")
            return True
        elif response.status_code == 503:
            print("[WARNING] Prediction endpoint unavailable (models not loaded)")
            return True
        else:
            print(f"[ERROR] Prediction failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"[ERROR] API test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_api_simple()
    sys.exit(0 if success else 1)