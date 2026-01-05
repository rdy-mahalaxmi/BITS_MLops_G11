#!/usr/bin/env python3
"""
Start MLflow UI to view pipeline runs
"""

import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

def start_mlflow_ui_background():
    """Start MLflow UI in background"""
    try:
        subprocess.run(["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"], 
                      cwd=os.getcwd(), capture_output=True)
    except Exception as e:
        print(f"MLflow UI error: {e}")

def start_mlflow_ui():
    """Start MLflow UI and open browser"""
    print("Starting MLflow UI in background...")
    
    # Start MLflow UI in background thread
    ui_thread = Thread(target=start_mlflow_ui_background, daemon=True)
    ui_thread.start()
    
    # Wait for UI to start
    time.sleep(3)
    
    # Open browser
    url = "http://localhost:5000"
    print(f"Opening MLflow UI at: {url}")
    webbrowser.open(url)
    
    print("MLflow UI is running in background.")
    print("Close this terminal to stop MLflow UI.")
    
    # Keep script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMLflow UI stopped.")

if __name__ == "__main__":
    start_mlflow_ui()