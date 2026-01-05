#!/usr/bin/env python3
"""
Alternative containerization using Podman (Docker alternative)
"""

import subprocess
import logging

logger = logging.getLogger(__name__)

def check_podman_availability():
    """Check if Podman is available"""
    try:
        result = subprocess.run(["podman", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def build_with_podman():
    """Build container image with Podman"""
    if not check_podman_availability():
        logger.warning("Neither Docker nor Podman available. Install one for containerization.")
        return True
    
    logger.info("Building with Podman...")
    result = subprocess.run(["podman", "build", "-t", "heart-disease-api:latest", "."], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Podman build successful")
        return True
    else:
        logger.error(f"Podman build failed: {result.stderr}")
        return False

def run_with_podman():
    """Run container with Podman"""
    if not check_podman_availability():
        return True
    
    # Stop existing container
    subprocess.run(["podman", "stop", "heart-disease-api-container"], 
                  capture_output=True)
    subprocess.run(["podman", "rm", "heart-disease-api-container"], 
                  capture_output=True)
    
    # Run new container
    result = subprocess.run([
        "podman", "run", "-d", "-p", "8000:8000", 
        "--name", "heart-disease-api-container", 
        "heart-disease-api:latest"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Podman container started successfully")
        return True
    else:
        logger.error(f"Podman run failed: {result.stderr}")
        return False