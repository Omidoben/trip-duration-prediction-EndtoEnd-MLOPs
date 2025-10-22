#!/usr/bin/env python
"""
Main training script for NYC Taxi Trip Duration Prediction

This is the main entry point to run the entire training pipeline.

It orchestrates:
1. Configuration loading
2. Data preparation
3. MLflow setup
4. Baseline model training
5. Hyperparameter tuning
6. Model comparison

Usage:
    python scripts/train.py

Output:
    - Trained models in models/
    - MLflow database in mlflow.db
    - Results comparison in models/model_comparison.csv
"""

import sys
from pathlib import Path

# Add project root to Python path so we can import src/
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

# Import custom modules from src/
from src.utils.config_loader import load_config
from src.data.load_data import load_train_val_data
from src.data.preprocessing import prepare_features
from src.models.train import (
    setup_mlflow,
    train_model
)


def validate_data(train_df, val_df, feature_cols, target_col):
    """
    Validate that data is loaded correctly
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        feature_cols: List of feature column names
        target_col: Target column name
        
    Raises:
        ValueError: If data is invalid
    """
    # Check data is not empty
    if train_df.empty:
        raise ValueError("Training data is empty!")
    if val_df.empty:
        raise ValueError("Validation data is empty!")
    
    # Check all features exist
    missing_in_train = set(feature_cols) - set(train_df.columns)
    missing_in_val = set(feature_cols) - set(val_df.columns)
    
    if missing_in_train:
        raise ValueError(f"Missing features in training data: {missing_in_train}")
    if missing_in_val:
        raise ValueError(f"Missing features in validation data: {missing_in_val}")
    
    # Check target exists
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data")
    if target_col not in val_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in validation data")
    
    print("Data validation passed")


def main():
    """
    Main training pipeline execution
    
    Flow:
        1. Load and validate config
        2. Setup MLflow
        3. Load and prepare data
        4. Train baseline models
        5. Tune hyperparameters
        6. Compare all models
        7. Save results
    """
    
    print("NYC TAXI TRIP DURATION PREDICTION - TRAINING PIPELINE")
    
    try:
        print("\n[1/6] Loading configuration...")
        
        config_path = "configs/config.yaml"
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        config = load_config(config_path)
        
        data_config = config["data"]
        mlflow_config = config["mlflow"]
        training_config = config["training"]
        models_config = config["models"]
        
        print(f"Experiment: {mlflow_config['experiment_name']}")
        print(f"CV folds: {training_config['cv_folds']}")
        print(f"Random state: {training_config['random_state']}")
        


        print("\n[2/6] Setting up MLflow...")
        
        setup_mlflow(
            tracking_uri=mlflow_config["tracking_uri"],
            experiment_name=mlflow_config["experiment_name"]
        )
        
        

        print("[3/6] Loading data...")
        
        # Verify data files exist
        train_file = Path(data_config["raw_dir"]) / data_config["train_file"]
        val_file = Path(data_config["raw_dir"]) / data_config["val_file"]
        
        if not train_file.exists():
            raise FileNotFoundError(f"Training data not found: {train_file}")
        if not val_file.exists():
            raise FileNotFoundError(f"Validation data not found: {val_file}")
        
        # Load data using config
        train_df, val_df = load_train_val_data(config)
        
        print(f"        Train shape: {train_df.shape}")
        print(f"        Validation shape: {val_df.shape}")
        
        # Prepare features from config
        feature_cols = (
            data_config["categorical_features"] + 
            data_config["numerical_features"]
        )
        target_col = data_config["target"]
        
        print(f"      Features: {len(feature_cols)}")
        print(f"      Target: {target_col}")
        
        # Validate data
        validate_data(train_df, val_df, feature_cols, target_col)
        
        # Prepare X and y
        X_train, y_train = prepare_features(train_df, feature_cols, target_col)
        X_val, y_val = prepare_features(val_df, feature_cols, target_col)
        
        
        # Dictionary to store all results for comparison
        results = {}
        
      

        print("\n[4/6] Training baseline models...")
        
        # Linear regression - simple baseline model
        if models_config["linear_regression"]["enabled"]:
            print("\n  Linear Regression (baseline)")
            _, val_metrics = train_model(
                model=LinearRegression(),
                model_name="linear_regression",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                categorical_features=data_config["categorical_features"],
                numerical_features=data_config["numerical_features"]
            )
            results["linear_regression"] = val_metrics["rmse"]
        
        # LASSO - L1 regularization baseline
        if models_config["lasso"]["enabled"]:
            print("\n  Lasso (baseline, alpha=0.001)")
            _, val_metrics = train_model(
                model=Lasso(alpha=0.001, max_iter=1000),
                model_name="lasso_baseline",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                categorical_features=data_config["categorical_features"],
                numerical_features=data_config["numerical_features"]
            )
            results["lasso_baseline"] = val_metrics["rmse"]
        
        # RIDGE - L2 regularization baseline
        if models_config["ridge"]["enabled"]:
            print("\n  Ridge (baseline, alpha=0.5)")
            _, val_metrics = train_model(
                model=Ridge(alpha=0.5),
                model_name="ridge_baseline",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                categorical_features=data_config["categorical_features"],
                numerical_features=data_config["numerical_features"]
            )
            results["ridge_baseline"] = val_metrics["rmse"]
        
        # Random forest
        if models_config["random_forest"]["enabled"]:
            print("\n Random Forest (baseline, n_estimators = 200)")
            _, val_metrics = train_model(
                model=RandomForestRegressor(n_estimators=100, max_depth=10, 
                                            min_samples_split=8, min_samples_leaf=5),
                model_name="random_forest",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                categorical_features=data_config["categorical_features"],
                numerical_features=data_config["numerical_features"]
            )
            results["random_forest_baseline"] = val_metrics["rmse"]
        


        print("\n[6/6] Comparing models...")
        print("\nFINAL RESULTS - MODEL COMPARISON")
        
        # Create results dataframe
        results_df = pd.DataFrame(
            list(results.items()),
            columns=['Model', 'Validation RMSE']
        )
        results_df = results_df.sort_values('Validation RMSE').reset_index(drop=True)
        
        # Display table
        print("\n")
        print(results_df.to_string(index=False))
        
        # Best model
        best_model = results_df.iloc[0]['Model']
        best_rmse = results_df.iloc[0]['Validation RMSE']
        
        print(f"BEST MODEL: {best_model.upper()}")
        print(f"   RMSE: {best_rmse:.2f} minutes")
        
        # Save comparison results
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        results_file = models_dir / 'model_comparison.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
        

        print(" TRAINING COMPLETE!")
        print("\n📊 View detailed results in MLflow UI:")
        print("\n📁 Models saved in: models/")
        print(f"\n📋 Comparison results: {results_file}")
        print()
        
    except FileNotFoundError as e:
        print(f"\n ERROR: {e}")
        sys.exit(1)
        
    except ValueError as e:
        print(f"\n ERROR: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()