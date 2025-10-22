# NYC Taxi Trip Duration Prediction
MLOps project to predict NYC taxi trip durations using machine learning.

## 📊 Project Overview
Predict trip duration in minutes based on:
- Pickup/dropoff locations
- Trip distance
- Passenger count
- Time features (hour, day of week)

## Tech Stack
- Python 3.x
- Scikit-learn (ML models)
- MLflow (experiment tracking)
- Pandas, NumPy (data processing)
- Jupyter (exploration)


## Getting Started
### Installation
1. Create environment:
```bash
conda create -n mlops python=3.10
conda activate mlops

2. Install dependencies:
    pip install -r requirements.txt

3. Download data:
Visit NYC TLC Trip Data
Download yellow_tripdata_2021-01.parquet and yellow_tripdata_2021-02.parquet
Place in data/raw/

Training Models
    python scripts/train.py

View Experiments
    mlflow ui
    Open browser: http://localhost:5000

📈 Models
Linear Regression (baseline)
Lasso Regression (L1 regularization)
Ridge Regression (L2 regularization)
Random Forest (ensemble)

🎯 Features
Categorical:

PU_DO (pickup-dropoff combination)
pickup_hour
pickup_dayofweek
payment_type

Numerical:
trip_distance
passenger_count

📊 Metrics
RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
R² Score

📝 Workflow
1. Exploration → notebooks/
2. Development → src/ (reusable code)
3. Execution → scripts/ (run training)
4. Tracking → MLflow UI (compare models)

🤝 Contributing
This is a learning project following MLOps best practices.

📚 Resources
MLOps Zoomcamp
MLflow Documentation

👤 Author
[Omido Benard]
