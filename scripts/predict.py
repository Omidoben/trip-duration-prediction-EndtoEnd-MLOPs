"""
Prediction Script - Make predictions on new NYC taxi trip data
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pickle
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

from src.utils.config_loader import load_config
from src.data.preprocessing import prepare_features


def load_model(model_path="models/ridge_baseline.pkl"):
    """Load trained Ridge model"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    return model


def prepare_input_data(input_file, config):
    """
    Prepare input data for prediction
    
    Expects CSV or Parquet with columns:
    - PULocationID, DOLocationID, payment_type
    - trip_distance, passenger_count
    - tpep_pickup_datetime (for time features)
    """
    data_config = config["data"]
    
    # Load data
    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
    elif input_file.endswith(".parquet"):
        df = pd.read_parquet(input_file)
    else:
        raise ValueError("File must be CSV or Parquet")
    
    print(f"Loaded {len(df)} records from {input_file}")
    
    # Validate required columns
    required_cols = ['PULocationID', 'DOLocationID', 'payment_type', 
                     'trip_distance', 'passenger_count', 'lpep_pickup_datetime']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Convert to string and create PU_DO
    categorical_cols = ["PULocationID", "DOLocationID", "payment_type"]
    df[categorical_cols] = df[categorical_cols].astype(str)
    df["PU_DO"] = df["PULocationID"] + '_' + df["DOLocationID"]
    
    # Add time features
    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["lpep_pickup_datetime"].dt.dayofweek
    
    # Select features
    feature_cols = (
        data_config["categorical_features"] + 
        data_config["numerical_features"]
    )
    
    X = df[feature_cols].copy()
    
    # Store original data for output
    original_data = df[['PULocationID', 'DOLocationID', 'trip_distance', 
                        'passenger_count', 'lpep_pickup_datetime']].copy()
    
    return X, original_data


def make_predictions(model, X):
    """Make predictions"""
    predictions = model.predict(X)
    return predictions


def format_output(predictions, original_data, format_type="csv"):
    """Format predictions for output"""
    output_df = original_data.copy()
    output_df["predicted_duration"] = predictions
    output_df["prediction_timestamp"] = datetime.now().isoformat()
    
    return output_df


def main():
    parser = argparse.ArgumentParser(
        description="Make predictions on new taxi trip data"
    )
    parser.add_argument('--input', required=True, 
                       help='Input file (CSV or Parquet)')
    parser.add_argument('--output', default='predictions.csv',
                       help='Output file for predictions (default: predictions.csv)')
    parser.add_argument('--model', default='models/ridge_baseline.pkl',
                       help='Path to trained model')
    parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                       help='Output format')
    
    args = parser.parse_args()
    
    try:
        print("PREDICTION SCRIPT - NYC TAXI TRIP DURATION")
        
        # Load configuration
        print("\n[1/4] Loading configuration...")
        config = load_config('configs/config.yaml')
        print("Config loaded")
        
        # Load model
        print("\n[2/4] Loading model...")
        model = load_model(args.model)
        print(f"Model loaded: {args.model}")
        
        # Prepare data
        print("\n[3/4] Preparing input data...")
        X, original_data = prepare_input_data(args.input, config)
        print(f"Data prepared: {X.shape}")
        
        # Make predictions
        print("\n[4/4] Making predictions...")
        predictions = make_predictions(model, X)
        print(f"Predictions made: {len(predictions)} records")
        
        # Format and save output
        output_df = format_output(predictions, original_data, args.format)
        
        if args.format == 'csv':
            output_df.to_csv(args.output, index=False)
        else:  # json
            output_df.to_json(args.output, orient='records', indent=2)
        
        
        print(f"\nPredictions Summary:")
        print(f"  Count: {len(predictions)}")
        print(f"  Mean duration: {predictions.mean():.2f} minutes")
        print(f"  Min duration: {predictions.min():.2f} minutes")
        print(f"  Max duration: {predictions.max():.2f} minutes")
        print(f"  Std dev: {predictions.std():.2f} minutes")
        
        print(f"\nOutput saved to: {args.output}")
        print(f"Format: {args.format}")
        
        # Display first few predictions
        print(f"\nFirst 5 predictions:")
        print(output_df[['trip_distance', 'passenger_count', 
                         'predicted_duration']].head())
        
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
