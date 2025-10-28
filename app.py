"""
Flask API for NYC Taxi Trip Duration Prediction
"""
import sys
import os
from pathlib import Path
import pickle
from datetime import datetime

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

from src.utils.config_loader import load_config

app = Flask(__name__)

# global variables to store model and config
model = None
config = None

# Base directory (important fix for Render)
BASE_DIR = Path(__file__).resolve().parent


def load_model_config():
    """
    Load model and config
    """
    global model, config

    try:
        config_path = BASE_DIR / "configs" / "config.yaml"
        model_path = BASE_DIR / "models" / "ridge_baseline.pkl"

        print(f"Looking for config at: {config_path}")
        print(f"Looking for model at: {model_path}")

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        config = load_config(config_path)
        print("Config loaded successfully")

        with open(model_path, "rb") as f_out:
            model = pickle.load(f_out)

        print(f"Model loaded from {model_path}")
        print("Model and config initialized")

        print("Current working directory:", os.getcwd())
        print("Files in BASE_DIR:", os.listdir(BASE_DIR))

    except Exception as e:
        print(f"Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.route("/", methods=["GET"])
def home():
    """Serve the main website"""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": "ridge_baseline",
        "version": "1.0"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Make predictions on taxi trip data"""
    try:
        data = request.get_json()

        if not data or "trips" not in data:
            return jsonify({
                "error": "Invalid input. Expected json with 'trips' key"
            }), 400

        trips = data["trips"]

        if not isinstance(trips, list) or len(trips) == 0:
            return jsonify({
                "error": "Trips must be a non empty list"
            }), 400

        # convert to dataframe
        df = pd.DataFrame(trips)

        # validate required columns
        if config is None or model is None:
            return jsonify({
                "error": "Model not initialized"
            }), 500

        data_config = config["data"]

        required_cols = [
            "PULocationID", "DOLocationID", "payment_type",
            "trip_distance", "passenger_count", "lpep_pickup_datetime"
        ]

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return jsonify({
                "error": f"Missing required columns: {missing}"
            }), 400

        # Prepare features
        categorical_cols = ["PULocationID", "DOLocationID", "payment_type"]
        df[categorical_cols] = df[categorical_cols].astype(str)
        df["PU_DO"] = df["PULocationID"] + '_' + df["DOLocationID"]

        df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
        df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour
        df["pickup_dayofweek"] = df["lpep_pickup_datetime"].dt.dayofweek

        feature_cols = (
            data_config["categorical_features"] +
            data_config["numerical_features"]
        )
        X = df[feature_cols].copy()

        predictions = model.predict(X)

        response = {
            "predictions": [
                {
                    "trip_distance": float(df.iloc[i]['trip_distance']),
                    "passenger_count": int(df.iloc[i]['passenger_count']),
                    "predicted_duration_minutes": float(predictions[i]),
                    "pickup_datetime": df.iloc[i]['lpep_pickup_datetime'].isoformat()
                }
                for i in range(len(predictions))
            ],
            "summary": {
                "num_predictions": len(predictions),
                "mean_duration": float(predictions.mean()),
                "min_duration": float(predictions.min()),
                "max_duration": float(predictions.max()),
                "timestamp": datetime.now().isoformat()
            }
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/info", methods=["GET"])
def model_info():
    """Get model information"""
    if config is None:
        return jsonify({
            "error": "Model not initialized"
        }), 500

    return jsonify({
        "model_type": "Ridge Regression",
        "model_file": "ridge_baseline.pkl",
        "features": {
            "categorical": config["data"]["categorical_features"],
            "numerical": config["data"]["numerical_features"]
        },
        "description": "Predicts NYC taxi trip duration in minutes"
    }), 200


load_model_config()

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', 8000)))
