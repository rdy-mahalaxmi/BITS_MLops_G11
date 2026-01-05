# Heart Disease Prediction MLOps Pipeline

A complete MLOps pipeline for heart disease prediction using machine learning, featuring automated CI/CD, containerization, and monitoring.

##  Architecture Overview

```
├── src/                    # Source code
│   ├── data_acquisition.py # Data download and EDA
│   ├── model_training.py   # Feature engineering and training
│   └── api.py             # FastAPI serving application
├── tests/                 # Unit tests
├── docker/               # Docker configurations
├── k8s/                  # Kubernetes manifests
├── monitoring/           # Prometheus/Grafana configs
├── .github/workflows/    # CI/CD pipeline
└── requirements.txt      # Dependencies
```

##  Quick Start

### Option 1: Run Complete Pipeline
```bash
python run_pipeline.py
```

### Option 2: Manual Steps

1. **Setup Environment**
```bash
pip install -r requirements.txt
mkdir data models plots mlruns
```

2. **Data Acquisition & EDA**
```bash
cd src && python data_acquisition.py
```

3. **Model Training**
```bash
cd src && python model_training.py
```

4. **Run API**
```bash
cd src && python api.py
```

5. **Test API**
```bash
curl -X POST \"http://localhost:8000/predict\" \
     -H \"Content-Type: application/json\" \
     -d '{
       \"age\": 50,
       \"sex\": 1,
       \"chest_pain\": 2,
       \"resting_bp\": 120,
       \"cholesterol\": 200,
       \"fasting_bs\": 0,
       \"resting_ecg\": 1,
       \"max_hr\": 150,
       \"exercise_angina\": 0,
       \"oldpeak\": 1.0,
       \"st_slope\": 1
     }'
```

##  Docker Deployment

### Build and Run
```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

### Using Docker Compose (with monitoring)
```bash
docker-compose up -d
```

Access:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

##  Kubernetes Deployment

```bash
kubectl apply -f k8s/deployment.yaml
kubectl port-forward service/heart-disease-api-service 8000:80
```

##  Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

##  MLflow Tracking

View experiments:
```bash
mlflow ui
```
Access at: http://localhost:5000

##  Monitoring

The API includes:
- Request/response logging
- Performance metrics
- Health checks
- Prometheus metrics endpoint

##  Assignment Requirements Coverage

###  1. Data Acquisition & EDA
- **Dataset**: Heart disease dataset with download script
- **Preprocessing**: Missing value handling, feature encoding
- **EDA**: Professional visualizations (histograms, correlation heatmaps, class balance)
- **Files**: `src/data_acquisition.py`

###  2. Feature Engineering & Model Development
- **Features**: Scaling and encoding pipeline
- **Models**: Logistic Regression and Random Forest
- **Tuning**: GridSearchCV with cross-validation
- **Metrics**: Accuracy, precision, recall, ROC-AUC
- **Files**: `src/model_training.py`

###  3. Experiment Tracking
- **MLflow**: Complete experiment tracking
- **Logging**: Parameters, metrics, artifacts, plots
- **Comparison**: Multiple model runs tracked
- **Files**: Integrated in `src/model_training.py`

###  4. Model Packaging & Reproducibility
- **Format**: MLflow and pickle model saving
- **Dependencies**: Clean `requirements.txt`
- **Pipeline**: Reusable preprocessing transformers
- **Files**: `models/`, `requirements.txt`

###  5. CI/CD Pipeline & Testing
- **Tests**: Comprehensive unit tests with pytest
- **Pipeline**: GitHub Actions workflow
- **Steps**: Linting, testing, model training
- **Artifacts**: Automated logging and storage
- **Files**: `.github/workflows/mlops-pipeline.yml`, `tests/`

###  6. Model Containerization
- **Container**: Docker with FastAPI
- **Endpoint**: `/predict` with JSON input/output
- **Features**: Health checks, logging, confidence scores
- **Files**: `Dockerfile`, `src/api.py`

###  7. Production Deployment
- **Platform**: Kubernetes deployment ready
- **Manifests**: Complete K8s configurations
- **Exposure**: LoadBalancer and Ingress
- **Verification**: Health and prediction endpoints
- **Files**: `k8s/deployment.yaml`

###  8. Monitoring & Logging
- **Logging**: Structured API request/response logging
- **Monitoring**: Prometheus + Grafana integration
- **Metrics**: Performance and health metrics
- **Files**: `monitoring/`, `docker-compose.yml`

## 🔧 API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /predict` - Heart disease prediction
- `GET /docs` - Interactive API documentation
- `GET /metrics` - Monitoring metrics

##  Model Performance

The pipeline trains and compares:
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble method with hyperparameter tuning

Models are evaluated using:
- Cross-validation (5-fold)
- ROC-AUC score
- Precision, Recall, Accuracy
- Confidence intervals

##  Development

### Code Quality
- **Linting**: flake8
- **Formatting**: black
- **Testing**: pytest with coverage
- **Type Hints**: pydantic models

### Continuous Integration
- Automated testing on push/PR
- Model training on main branch
- Docker image building
- Artifact storage


##  Troubleshooting

### Common Issues

1. **Models not found**: Run `python src/model_training.py` first
2. **Port conflicts**: Change port in `src/api.py` or stop conflicting services
3. **Docker issues**: Ensure Docker is installed in local, running and has sufficient resources
4. **Permission errors**: Check file permissions and user access

### Logs Location
- Application logs: Console output
- MLflow logs: `mlruns/` directory
- Docker logs: `docker logs heart-disease-api-container`

##  Dependencies

Key libraries:
- **ML**: scikit-learn, pandas, numpy
- **API**: FastAPI, uvicorn, pydantic
- **Tracking**: MLflow
- **Testing**: pytest, pytest-cov
- **Visualization**: matplotlib, seaborn

##  License
This project is for educational purposes as part of MLOps coursework.

**Note**: This is a complete MLOps implementation covering all assignment requirements with production-ready code, comprehensive testing, and full automation.
