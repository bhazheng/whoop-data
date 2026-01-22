# Whoop AI Coach: Daily Intensity Recommender

A production-ready machine learning project to recommend daily training intensity based on Whoop fitness metrics. The AI Coach predicts recovery status to guide optimal workout strain.

## Problem Description

This project addresses the challenge of **automatically classifying physical activities** based on physiological and demographic data collected from Whoop fitness trackers. The model predicts whether a user is engaged in a specific activity type using features such as:

- Demographic information (age, height, weight, gender, BMI)
- Heart rate metrics (max, average, resting)
- Activity metrics (calories burned, distance, steps)
- Sleep and recovery data

The solution enables automated activity recognition which can be used for:
- Personalized fitness coaching
- Activity tracking validation
- Health insights and recommendations

## Project Structure

```
.
├── app/
│   └── main.py                # Flask API for model serving
├── code/
│   └── train.py               # Training script
├── notebooks/
│   └── notebook.ipynb         # EDA and model selection analysis
├── output/
│   ├── model/                 # Saved models
│   └── prediction/            # Prediction outputs
├── kubernetes/                # Kubernetes manifests
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
├── .dockerignore              # Docker build exclusions
├── .gitignore                 # Git exclusions
├── whoop_fitness_dataset_100k.csv    # Training dataset (21MB)
└── README.md                  # Project documentation
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd whoop-data
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Exploratory Data Analysis

Explore the dataset and comprehensive model comparison process:

```bash
jupyter notebook notebooks/notebook.ipynb
```

The notebook includes:
- **Data Loading & Preparation**: Robust path handling and data quality checks
- **Target Generation**: 3-class multiclass target (Rest/Light, Moderate, Push Hard)
- **Feature Engineering**: BMI, Training Load, Sleep Quality Score, HRV/RHR Ratio, and more
- **Exploratory Data Analysis**: Target distribution, numerical/categorical features, correlations
- **Feature Importance**: Random Forest-based feature importance analysis
- **Baseline Models**: Train all 3 models (Logistic Regression, LightGBM, MLP) with default parameters
- **Hyperparameter Tuning**: Optimize all 3 models using RandomizedSearchCV
- **Tuned Model Evaluation**: Evaluate all 3 models after tuning
- **Before/After Comparison**: Side-by-side comparison of baseline vs tuned performance with visualizations

**Key Features:**
- Trains **all 3 algorithms**, not just the best one
- Shows clear **before/after tuning comparison** for each model
- Displays confusion matrices for all models
- Visualizes performance improvements from hyperparameter optimization

### 2. Train the Model

Train the final model with default parameters:

```bash
python code/train.py
```

Or customize hyperparameters:

```bash
python code/train.py --max_depth 15 --n_estimators 200
```

**Arguments:**
- `--max_depth`: Maximum depth for Random Forest and XGBoost (default: 10)
- `--n_estimators`: Number of estimators for ensemble models (default: 100)
- `--data_path`: Path to dataset (default: `whoop_fitness_dataset_100k.csv`)

**Output:**
- `output/model/model_final.bin`: Serialized model and DictVectorizer

- `output/model/model_final.bin`: Serialized model and DictVectorizer

### 3. Standalone Prediction (CLI)

Run predictions without starting the server using the new validation script:

```bash
# Using a JSON file
python code/predict.py --file input.json

# Using a JSON string
python code/predict.py --input '{"age": 30, "height_cm": 180, "weight_kg": 75, ...}'
```

This script applies the exact same feature engineering (Age Bins, BMI, Lowercasing, Imputation) as the live service.

### 4. Run the Prediction Service

Start the Flask API server:

```bash
python app/main.py
```

The service will be available at `http://localhost:5000`

**API Endpoints:**

- **Health Check**: `GET /`
  ```bash
  curl http://localhost:5000/
  ```

- **Predict**: `POST /predict`
  ```bash
  curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "age": 30,
      "height_cm": 175,
      "weight_kg": 70,
      "gender": "M",
      "max_heart_rate": 180,
      "avg_heart_rate": 140,
      "resting_heart_rate": 60,
      "calories_burned": 500,
      "distance_km": 5.0,
      "steps": 7000
    }'
  ```

  **Response:**
  ```json
  {
    "activity_prediction": 1,
    "probability": 0.85,
    "model": "XGBoost"
  }
  ```

### 4. Docker Deployment

Build and run with Docker:

```bash
# Build the image
docker build -t whoop-classifier .

# Run the container
docker run -p 5000:5000 whoop-classifier
```

Access the service at `http://localhost:5000`

### 5. Cloud Deployment (Kubernetes)

Deploy to a Kubernetes cluster (e.g., local Minikube or cloud provider):

1. **Start Minikube** (if running locally)
   ```bash
   minikube start
   ```

2. **Load Docker image**
   ```bash
   # Build image inside Minikube environment
   eval $(minikube docker-env)
   docker build -t whoop-classifier:latest .
   ```

3. **Deploy to Kubernetes**
   ```bash
   # Apply manifests
   kubectl apply -f kubernetes/deployment.yaml
   kubectl apply -f kubernetes/service.yaml
   ```

4. **Access the Service**
   ```bash
   # Check status
   kubectl get pods
   
   # Get service URL (for Minikube)
   minikube service whoop-classifier-service --url
   ```

   The service will be scalable and managed by Kubernetes with automatic restarts and load balancing.

## Dataset

The project uses the **Whoop Fitness Dataset** (`whoop_fitness_dataset_100k.csv`), which contains 100,000 records of fitness activity data.

**Dataset Features:**
- Demographic: age, height, weight, gender
- Physiological: heart rate metrics, VO2 max
- Activity: calories, distance, steps, duration
- Recovery: sleep hours, recovery score

The dataset is included in the repository (21MB).

## Model Performance

The training pipeline evaluates three models:

| Model                 | Validation ROC AUC | Test ROC AUC |
|-----------------------|-------------------|--------------|
| Logistic Regression   | ~0.XX             | ~0.XX        |
| Random Forest         | ~0.XX             | ~0.XX        |
| **XGBoost** (final)   | **~0.XX**         | **~0.XX**    |

The best-performing model is automatically selected and retrained on the full training set before deployment.

## Development

### Running Tests

Train and validate the model:

```bash
python train.py --max_depth 10 --n_estimators 100
```

Test the prediction service locally:

```bash
python predict.py
# In another terminal:
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @sample_request.json
```

### Adding New Features

1. Update data preparation in `train.py` (e.g., `load_and_prepare_data()`)
2. Retrain the model
3. Update the prediction service input validation in `predict.py`

## Dependencies

Key dependencies:
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: ML models and preprocessing
- **xgboost**: Gradient boosting
- **flask**: Web service framework
- **gunicorn**: Production WSGI server

See `requirements.txt` for complete list with pinned versions.

## License

This project is for educational and research purposes.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## Contact

For questions or issues, please open an issue in the repository.
