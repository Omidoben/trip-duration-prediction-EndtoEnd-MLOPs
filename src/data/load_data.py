# Data Loading Module - Reusable data loading functions used everywhere

"""Data loading and cleaning functions"""
import pandas as pd
from pathlib import Path

def load_clean_data(filepath, min_duration,
                     max_duration, min_distance, max_distance, year=None, month=None):
    """
    Load and clean NYC taxi trip data
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    filepath = Path(filepath)

    if filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)

    elif filepath.suffix == ".csv":
        df = pd.read_csv(filepath)
        df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])
        df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])

    else:
        raise ValueError("File must be .parquet or .csv")
    
    # Trip duration
    df["duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60

    # Filter trip distance and duration 
    df = df[(df["duration"] >= min_duration) & (df["duration"] <= max_duration)]
    df =  df[(df["trip_distance"] >= min_distance) & (df["trip_distance"] <= max_distance)]

    # Filter by year and month if specified
    if year:
        df = df[df["lpep_pickup_datetime"].dt.year == year]
    if month:
        df = df[df["lpep_pickup_datetime"].dt.month == month]

    categorical_cols = ["PULocationID", "DOLocationID", "payment_type"]
    df[categorical_cols] = df[categorical_cols].astype(str)
    
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    return df


def add_time_features(df):
    """
    Add time-based features to dataframe
    """
    df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["lpep_pickup_datetime"].dt.dayofweek
    df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype(int)
    
    return df


def load_train_val_data(config):
    """
    Load both training and validation data using config

    """
    data_config = config["data"]
    
    # Build paths
    train_path = Path(data_config["raw_dir"]) / data_config["train_file"]
    val_path = Path(data_config["raw_dir"]) / data_config["val_file"]
    
    # Load data
    train_df = load_clean_data(
    train_path,
    year=2021,
    month=1,
    min_duration=data_config["min_duration"],
    max_duration=data_config["max_duration"],
    min_distance=data_config["min_distance"],
    max_distance=data_config["max_distance"]
    )
    
    val_df = load_clean_data(
    val_path,
    year=2021,
    month=2,
    min_duration=data_config["min_duration"],
    max_duration=data_config["max_duration"],
    min_distance=data_config["min_distance"],
    max_distance=data_config["max_distance"]
    )
    
    # Add time features
    train_df = add_time_features(train_df)
    val_df = add_time_features(val_df)
    
    return train_df, val_df





