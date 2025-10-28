# 🚕 NYC Taxi Trip Duration Prediction (MLOps Project)

A full end-to-end **MLOps project** to predict NYC taxi trip durations using machine learning.  
It includes data ingestion, preprocessing, model training, experiment tracking, and web deployment.

🔗 **Live Demo:** [https://trip-duration-prediction.onrender.com](https://trip-duration-prediction.onrender.com)

---

## 📊 Project Overview
The goal of this project is to predict taxi trip duration (in minutes) based on:
- Pickup and dropoff locations  
- Trip distance  
- Passenger count  
- Temporal features (hour and day of week)

The solution demonstrates **MLOps best practices**, from data preparation to production deployment.

---

## ⚙️ Tech Stack
| Category | Tools |
|-----------|--------|
| Language | Python 3.10 |
| ML & Data | Scikit-learn, Pandas, NumPy |
| Experiment Tracking | MLflow |
| Deployment | Flask, Gunicorn, Render |
| Others | YAML configs, Git, Docker (optional) |

---

## 🧠 Model Training Workflow
1. **Exploration** → `notebooks/`  
   Data exploration and feature engineering.  
2. **Development** → `src/`  
   Modular, reusable functions for data processing and model training.  
3. **Execution** → `scripts/`  
   Scripts for automated training and evaluation (`train.py`).  
4. **Tracking** → `MLflow UI`  
   Compare model runs, parameters, and metrics visually.  

---

## 🚀 Getting Started

### 1️⃣ Setup Environment
```bash
# Create virtual environment
conda create -n mlops python=3.10
conda activate mlops

# Install dependencies
pip install -r requirements.txt


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



🤝 Contributing
This is a learning project following MLOps best practices.

📚 Resources
MLOps Zoomcamp
MLflow Documentation

👤 Author
[Omido Benard]
