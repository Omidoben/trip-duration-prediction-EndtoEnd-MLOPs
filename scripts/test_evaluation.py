"""
Loads Ridge model and evaluates it on test data
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from src.utils.config_loader import load_config
from src.data.load_data import load_clean_data, add_time_features
from src.data.preprocessing import prepare_features


def main():
    print("Ridge Model - Test Evaluation")

    try:
        print("\n [1/4] Loading Configuration...")
        config = load_config("configs/config.yaml")
        data_config = config["data"]
        print("Config loaded")

        # Load test data
        print("\n [2/4] Loading test data")
        test_file = Path(data_config["raw_dir"]) / data_config["test_file"]

        if not test_file.exists():
            raise FileNotFoundError(f"Test data not found: {test_file}")
        
        test_df = load_clean_data(
            str(test_file),
            year=2021,
            month=3,
            min_duration=data_config["min_duration"],
            max_duration=data_config["max_duration"],
            min_distance=data_config["min_distance"],
            max_distance=data_config["max_distance"]
        )
        test_df=add_time_features(test_df)

        print(f"Test data loaded: {test_df.shape}")


        # Prepare features
        print("\n [3/4] Preparing features...")
        feature_cols = (data_config["categorical_features"] + data_config["numerical_features"])
        target_col = data_config["target"]

        X_test, y_test = prepare_features(test_df, feature_cols, target_col)
        print(f"Features prepared: {X_test.shape}")

        # Load ridge model
        print("\n [4/4] Loading ridge model and making predictions")
        model_path = Path("models/ridge_baseline.pkl")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, "rb") as f:
            ridge_model = pickle.load(f)
        print(f"Model loaded")

        # Make predictions
        y_pred = ridge_model.predict(X_test)
        print("Predictions made")

        # Metrics
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # results
        print(f"\nRMSE: {rmse:.2f} minutes")
        print(f"MAE:  {mae:.2f} minutes")
        print(f"R²:   {r2:.4f}")

        # Error analysis
        residuals = y_test - y_pred
        print(f"\nError Analysis:")
        print(f"  Mean residual: {residuals.mean():.2f} minutes")
        print(f"  Std residual:  {residuals.std():.2f} minutes")
        print(f"  Min residual:  {residuals.min():.2f} minutes")
        print(f"  Max residual:  {residuals.max():.2f} minutes")
        
        # Save results
        results_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'R²'],
            'Value': [rmse, mae, r2]
        })
        
        results_file = Path('models/ridge_test_results.csv')
        results_df.to_csv(results_file, index=False)
        print(f"Results saved: {results_file}")
        
        print("Evaluation Complete")
        
    except Exception as e:
        print(f"\nError : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
