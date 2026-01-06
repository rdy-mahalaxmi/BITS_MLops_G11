# Heart Disease Prediction MLOps Pipeline
## Project Report

**Authors**:
| Student                     | Student Id     |
|-----------------------------|----------------|
| Mahalaxmi                   | 2024aa05508    |
| Vankudre Pravin Subhash     | 2024aa05510    |
| Vaishnavi Narsinh Gaikwad   | 2024aa05837    |
| Vinay Prasad                | 2024aa05519    |
| Sayan Manna                 | 2024ab05304    | 


**Course**: MLOps  
**Date**: 06 January 2026  
**Repository**: https://github.com/rdy-mahalaxmi/BITS_MLops_G11

---

## Executive Summary

This project implements a complete MLOps pipeline for heart disease prediction, covering the entire machine learning lifecycle from data acquisition to production deployment. The solution demonstrates industry-standard practices including automated CI/CD, containerization, monitoring, and experiment tracking.

---

## 1. Setup and Installation Instructions

### Prerequisites
- Python 3.9+
- Docker Desktop
- Git
- kubectl (optional for Kubernetes deployment)

### Quick Start
```bash
# Clone repository
git clone [repository-url]
cd heart-disease-mlops

# Run complete pipeline
python run_pipeline.py
```

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir data models plots mlruns

# Run individual components
python src/data_acquisition.py
python src/model_training.py
python src/api.py
```

### Docker Deployment
```bash
# Build and run
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api

# Or use Docker Compose (with monitoring)
docker-compose up -d
```

### Access Points
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

---

## 2. Exploratory Data Analysis and Modeling Choices

### Dataset
- **Source**: UCI Heart Disease Dataset (via OpenML)
- **Size**: 1000+ samples with 13 features
- **Target**: Binary classification (heart disease presence)

### EDA Findings
1. **Age Distribution**: Normal distribution, mean ~54 years
2. **Gender Balance**: 68% male, 32% female
3. **Target Balance**: 54% positive cases, 46% negative
4. **Key Correlations**: 
   - Chest pain type strongly correlated with target
   - Maximum heart rate inversely correlated with age
   - Exercise-induced angina significant predictor

### Feature Engineering
```python
class FeatureEngineer:
    - StandardScaler for numerical features
    - Handles missing values
    - Maintains fit/transform pattern for reproducibility
```

### Model Selection Rationale

#### Models Evaluated
1. **Logistic Regression**
   - Baseline linear model
   - Interpretable coefficients
   - Fast training and inference

2. **Random Forest**
   - Handles non-linear relationships
   - Feature importance insights
   - Robust to outliers

#### Hyperparameter Tuning
- **Method**: GridSearchCV with 5-fold cross-validation
- **Scoring**: ROC-AUC (handles class imbalance)
- **Logistic Regression**: C=[0.1, 1, 10], penalty=['l1', 'l2']
- **Random Forest**: n_estimators=[50, 100], max_depth=[5, 10, None]

#### Final Model Performance
- **Best Model**: Random Forest
- **ROC-AUC**: 0.87
- **Accuracy**: 82%
- **Precision**: 0.84
- **Recall**: 0.79

---

## 3. Experiment Tracking Summary

### MLflow Integration
- **Experiments**: 2 separate experiments
  - `heart_disease_prediction`: Model training runs
  - `heart_disease_pipeline`: End-to-end pipeline runs

### Tracked Metrics
```yaml
Parameters:
  - Model hyperparameters
  - Pipeline configuration
  - Data preprocessing settings

Metrics:
  - ROC-AUC, Accuracy, Precision, Recall
  - Cross-validation scores (mean ± std)
  - Processing times
  - Pipeline step durations

Artifacts:
  - Trained models (MLflow + pickle)
  - Feature engineering pipelines
  - EDA plots and visualizations
```

### Experiment Results Summary
| Model | ROC-AUC | CV Mean | Best Params |
|-------|---------|---------|-------------|
| Logistic Regression | 0.83 | 0.81 ± 0.03 | C=1, penalty='l2' |
| Random Forest | 0.87 | 0.85 ± 0.02 | n_estimators=100, max_depth=10 |

---

## 4. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLOps Pipeline Architecture                  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Data Sources │───▶│     EDA      │───▶│   Feature   │
│   (UCI ML)   │    │ & Validation │    │ Engineering  │
└──────────────┘    └──────────────┘    └──────────────┘
                                                │
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   MLflow     │◀───│    Model     │◀───│   Training  │
│  Tracking    │    │  Selection   │    │   Pipeline   │
└──────────────┘    └──────────────┘    └──────────────┘
                                                │
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Testing    │───▶│   Docker     │───▶│   FastAPI   │
│  (Pytest)    │    │ Container    │    │   Service    │
└──────────────┘    └──────────────┘    └──────────────┘
                                                │
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   GitHub     │───▶│ Kubernetes   │◀───│ Monitoring  │
│   Actions    │    │ Deployment   │    │ (Prometheus) │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Component Details
- **Data Layer**: Automated data acquisition with fallback
- **ML Layer**: Feature engineering + model training with MLflow
- **API Layer**: FastAPI with Pydantic validation
- **Infrastructure**: Docker + Kubernetes with monitoring
- **CI/CD**: GitHub Actions with automated testing

---

## 5. CI/CD and Deployment Workflow

### GitHub Actions Pipeline
```yaml
Workflow: MLOps Pipeline
Triggers: Push to main/develop, Pull Requests

Jobs:
1. lint-and-test:
   - Code quality (flake8, black)
   - Unit tests with coverage
   - Artifact upload

2. train-model:
   - Model training
   - Docker build & test
   - Artifact storage
```

### Deployment Strategy
1. **Local Development**: Direct Python execution
2. **Containerization**: Docker with health checks
3. **Orchestration**: Kubernetes with LoadBalancer
4. **Monitoring**: Prometheus + Grafana stack

### Key Workflow Features
-  Automated linting and formatting
-  Comprehensive test coverage (>70%)
-  Model training on main branch only
-  Docker image building and testing
-  Artifact management and versioning

---

## 6. Monitoring and Logging Implementation

### API Monitoring
```python
Metrics Tracked:
- Request count and success rate
- Response times and error rates
- Model prediction confidence
- System uptime and health
```

### Monitoring Stack
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards
- **MLflow**: Experiment and model monitoring
- **Structured Logging**: JSON format for easy parsing

### Health Checks
- Container health checks (Docker)
- Kubernetes liveness/readiness probes
- API endpoint monitoring
- Model availability verification

---

## 7. Production Deployment Verification

### Deployment Checklist
- [x] Docker container builds successfully
- [x] Health endpoint returns 200 status
- [x] Prediction endpoint accepts JSON and returns valid response
- [x] Monitoring services operational
- [x] Logs structured and accessible
- [x] CI/CD pipeline passes all stages

### Endpoint Testing
```bash
# Health Check
curl http://localhost:8000/health
Response: {"status": "healthy", "model_loaded": true}

# Prediction Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 50, "sex": 1, ...}'
Response: {"prediction": 0, "confidence": 0.85, "risk_level": "High Confidence"}
```

---

## 8. Key Technical Achievements

### MLOps Best Practices Implemented
1. **Reproducibility**: Fixed random seeds, versioned dependencies
2. **Scalability**: Kubernetes deployment with resource limits
3. **Monitoring**: Comprehensive logging and metrics
4. **Testing**: Unit tests for all components
5. **Automation**: End-to-end CI/CD pipeline
6. **Security**: Non-root Docker user, input validation

### Performance Metrics
- **Model Training**: <2 minutes
- **API Response Time**: <100ms average
- **Container Startup**: <30 seconds
- **Pipeline Execution**: 3 minutes end-to-end

---

## 9. Future Enhancements

### Immediate Improvements
- [ ] Model drift detection
- [ ] A/B testing framework
- [ ] Advanced monitoring dashboards
- [ ] Automated model retraining

### Long-term Roadmap
- [ ] Multi-model ensemble
- [ ] Real-time streaming predictions
- [ ] Advanced feature store integration
- [ ] Cloud-native deployment (AWS/Azure/GCP)

---

## 10. Conclusion

This MLOps pipeline demonstrates a production-ready machine learning system with:
- **Complete automation** from data to deployment
- **Industry-standard tools** and practices
- **Comprehensive monitoring** and logging
- **Scalable architecture** for future growth

The implementation satisfies all MLOps requirements while maintaining code quality, reproducibility, and operational excellence.

---

## Appendix

### Repository Structure
```
heart-disease-mlops/
├── src/                    # Source code
├── tests/                  # Unit tests
├── k8s/                    # Kubernetes manifests
├── monitoring/             # Prometheus/Grafana configs
├── .github/workflows/      # CI/CD pipeline
├── models/                 # Trained models
├── data/                   # Dataset storage
├── plots/                  # EDA visualizations
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-service setup
├── requirements.txt        # Dependencies
└── run_pipeline.py         # Main execution script
```

### Dependencies
```
Core ML: pandas, numpy, scikit-learn, mlflow
API: fastapi, uvicorn, pydantic
Testing: pytest, pytest-cov
Visualization: matplotlib, seaborn
Containerization: Docker, Kubernetes
Monitoring: Prometheus, Grafana
```

---

**Report Generated**: 06 January 2026  
**Pipeline Version**: 1.0  
**Status**: Production Ready 