# 🏃‍♂️ Whoop AI Coach: Daily Intensity Recommender

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-green?logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Deployable-326CE5?logo=kubernetes&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

A production-ready machine learning system that predicts **recovery status** from Whoop fitness tracker data and recommends optimal daily training intensity.

---

## Problem Description

This project solves the challenge of **automated recovery status classification** to help athletes optimize their training load. Using physiological and activity metrics from Whoop fitness trackers, the AI Coach predicts one of three recovery states:

- 🔴 **Rest/Light** (0-33% recovery) - Take it easy, prioritize recovery
- 🟡 **Moderate** (33-67% recovery) - Balanced training appropriate
- 🟢 **Push Hard** (67-100% recovery) - Go all out, your body is ready

### Input Features
- **Physiological Metrics**: Heart Rate Variability (HRV), Resting Heart Rate, Max HR
- **Sleep Data**: Sleep hours, efficiency, REM, deep sleep, wake-ups
- **Activity Metrics**: Day strain, activity strain, calories burned, steps, distance
- **Demographic Data**: Age, gender, BMI, fitness level
- **Derived Features**: Training load, sleep quality score, HRV/RHR ratio

### Business Value
- **Optimize Performance**: Train hard when recovered, rest when needed
- **Prevent Overtraining**: Data-driven intensity recommendations
- **Personalized Coaching**: Tailored to individual recovery patterns
- **Track Progress**: Monitor recovery trends over time

---

## Architecture

```
┌─────────────────┐
│  Raw Data       │
│  (100k records) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature Eng.   │
│  (BMI, Load,    │
│   Sleep Score)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │
│  - Logistic Reg │
│  - LightGBM     │
│  - MLP          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Files    │
│  (model_*.bin)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Flask API      │
│  (Gunicorn)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Docker Image   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Kubernetes     │
│  (Production)   │
└─────────────────┘
```

**Pipeline Flow:**
1. **Data** → Whoop fitness dataset (100k samples)
2. **Train** → 3 multiclass models with hyperparameter tuning
3. **Save** → Individual model binaries (`model_lgbm.bin`, `model_lr.bin`, `model_mlp.bin`)
4. **Serve** → Flask REST API with model selection
5. **Deploy** → Docker containerization → Kubernetes orchestration

---

## Project Structure

```
whoop-data/
├── app/
│   └── main.py              # Flask API server
├── code/
│   ├── train.py             # Model training pipeline
│   └── predict.py           # CLI prediction tool
├── notebooks/
│   └── notebook.ipynb       # EDA & model comparison
├── data/
│   └── whoop_fitness_dataset_100k.csv
├── output/
│   ├── model/               # Saved models (*.bin)
│   └── prediction/          # Prediction outputs
├── kubernetes/
│   ├── deployment.yaml      # K8s deployment
│   └── service.yaml         # K8s service
├── Pipfile                  # Dependency management
├── Makefile                 # Automation commands
├── Dockerfile               # Container definition
├── docker-compose.yml       # Local orchestration
└── README.md
```

---

## Quick Start (Using Makefile)

### Prerequisites
- Python 3.9+
- Docker (for containerized deployment)
- Make (for automation)

### 1. Setup Environment
```bash
make setup
```
This installs `pipenv` and all dependencies.

### 2. Train Models
```bash
make train
```
Trains all 3 models (Logistic Regression, LightGBM, MLP) and saves them to `output/model/`.

### 3. Run API Service (Docker)
```bash
make run
```
Builds Docker image and starts the Flask API on `http://localhost:9696`.

### 4. Test API
```bash
make test
```
Sends a test prediction request using `input.json`.

### Other Useful Commands
```bash
make notebook   # Start Jupyter notebook
make clean      # Remove generated files
make all        # Full workflow: setup → train → run
```

---

## 💻 Usage

### Option 1: API Prediction (Recommended)

**Start the service:**
```bash
docker run -p 9696:9696 whoop-coach
```

**Make a prediction:**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 28,
    "gender": "male",
    "weight_kg": 75,
    "height_cm": 180,
    "fitness_level": "intermediate",
    "sleep_hours": 7.5,
    "hrv": 65,
    "resting_heart_rate": 58,
    "day_strain": 12.5,
    "model": "lgbm"
  }'
```

**Response:**
```json
{
  "recommendation": "Push Hard",
  "status_code": 2,
  "probability": 0.87,
  "model_used": "lgbm"
}
```

**Select Different Model:**
```bash
curl -X POST http://localhost:9696/predict?model=lr ...  # Logistic Regression
curl -X POST http://localhost:9696/predict?model=mlp ... # Neural Network
```

### Option 2: CLI Prediction
```bash
pipenv run python code/predict.py --model lgbm --input input.json
```

### Option 3: Programmatic Use
```python
import pickle
import pandas as pd

# Load model
with open('output/model/model_lgbm.bin', 'rb') as f:
    data = pickle.load(f)
    pipeline = data['pipeline']

# Prepare features (same as train.py)
features = {...}  # Your feature dict

# Predict
prediction = pipeline.predict([features])[0]
# 0=Rest/Light, 1=Moderate, 2=Push Hard
```

---

## Model Performance

All 3 models are trained with hyperparameter tuning:

| Model               | Baseline Acc | Tuned Acc | ROC AUC | Best Use Case           |
|---------------------|--------------|-----------|---------|-------------------------|
| **LightGBM**        | 0.8534       | 0.8689    | 0.9356  | Best overall (default)  |
| **Logistic Reg**    | 0.8245       | 0.8367    | 0.9089  | Interpretability        |
| **MLP (Neural Net)**| 0.8412       | 0.8545    | 0.9267  | Complex patterns        |

**Training Details:**
- Dataset: 100,000 samples
- Train/Test Split: 80/20
- Tuning: RandomizedSearchCV (3-fold CV)
- Metrics: Accuracy, ROC AUC (multiclass weighted)

See `notebooks/notebook.ipynb` for complete before/after tuning analysis.

---

## Deployment

### Local Docker
```bash
docker build -t whoop-coach .
docker run -p 9696:9696 whoop-coach
```

### Docker Compose
```bash
docker-compose up
```

### Kubernetes
```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

**Access the service:**
```bash
kubectl port-forward svc/whoop-coach-service 9696:80
```

---

## Notebooks

The Jupyter notebook provides a comprehensive analysis:

1. **Data Loading & Preparation** - Robust data handling
2. **EDA** - Target distribution, correlations, feature analysis
3. **Feature Importance** - Random Forest feature rankings
4. **Baseline Training** - All 3 models with default parameters
5. **Hyperparameter Tuning** - Optimized parameters for each model
6. **Tuned Evaluation** - Performance after optimization
7. **Before/After Comparison** - Visual performance improvements

**Run the notebook:**
```bash
make notebook
# or
pipenv run jupyter notebook notebooks/notebook.ipynb
```

---

## Development

### Install Dependencies
```bash
pipenv install --dev
```

### Train with Custom Parameters
```bash
pipenv run python code/train.py
```

### Run Tests
```bash
make test
```

### Clean Generated Files
```bash
make clean
```

---

## Dependencies

**Core Libraries:**
- `pandas` - Data manipulation
- `scikit-learn` - ML algorithms and preprocessing
- `lightgbm` - Gradient boosting
- `xgboost` - Gradient boosting (alternative)
- `numpy` - Numerical computing

**API & Deployment:**
- `flask` - Web framework
- `gunicorn` - Production WSGI server

**Visualization:**
- `matplotlib`, `seaborn` - Plotting and visualization

**Development:**
- `jupyter`, `ipykernel` - Interactive notebooks

See `Pipfile` for complete dependency list.

---

## Workflow Summary

```bash
# 1. Setup environment
make setup

# 2. Train models (creates 3 .bin files)
make train

# 3. Explore notebook (optional)
make notebook

# 4. Deploy via Docker
make run

# 5. Test API
make test
```

---

## API Endpoints

### Health Check
```bash
GET /
```
Returns service status and available models.

### Single Prediction
```bash
POST /predict?model=lgbm
Content-Type: application/json

{
  "age": 28,
  "gender": "male",
  ...
}
```

### Batch Prediction
```bash
POST /predict_batch?model=lgbm
Content-Type: application/json

{
  "instances": [
    {...},
    {...}
  ]
}
```

---

## Acknowledgments

- Dataset: Whoop Fitness Tracker Data
- ML Stack: scikit-learn, LightGBM
- Deployment: Docker, Kubernetes, Flask

---

