"""
Hyperparameter Tuning Script for NYC Taxi Trip Duration Prediction

This script performs hyperparameter tuning for baseline models:
- Lasso: Grid Search on alpha values
- Ridge: Grid Search on alpha values
- Random Forest: Random Search on multiple parameters
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

# Import custom modules
from src.utils.config_loader import load_config
from src.data.load_data import load_train_val_data
from src.data.preprocessing import prepare_features
from src.models.train import (
    setup_mlflow,
    tune_model_grid_search,
    tune_model_random_search
)


def main():
    """
    Main hyperparameter tuning pipeline
    
    Flow:
        1. Load configuration
        2. Setup MLflow
        3. Load and prepare data
        4. Tune Lasso with Grid Search
        5. Tune Ridge with Grid Search
        6. Tune Random Forest with Random Search
        7. Compare all results
        8. Save comparison
    """
    
    print("NYC TAXI TRIP DURATION - HYPERPARAMETER TUNING")
    
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
        
        print(f"      Experiment: {mlflow_config['experiment_name']}")
        print(f"      CV folds: {training_config['cv_folds']}")
        
        
        print("\n[2/6] Setting up MLflow...")
        
        setup_mlflow(
            tracking_uri=mlflow_config["tracking_uri"],
            experiment_name=mlflow_config["experiment_name"]
        )
        
        
        print("\n[3/6] Loading data...")
        
        # Verify data files exist
        train_file = Path(data_config["raw_dir"]) / data_config["train_file"]
        val_file = Path(data_config["raw_dir"]) / data_config["val_file"]
        
        if not train_file.exists():
            raise FileNotFoundError(f"Training data not found: {train_file}")
        if not val_file.exists():
            raise FileNotFoundError(f"Validation data not found: {val_file}")
        
        # Load data
        print("      Loading data files (this may take a moment)...")
        train_df, val_df = load_train_val_data(config)
        
        print(f"      Train shape: {train_df.shape}")
        print(f"      Validation shape: {val_df.shape}")
        
        # Prepare features
        feature_cols = (
            data_config["categorical_features"] + 
            data_config["numerical_features"]
        )
        target_col = data_config["target"]
        
        X_train, y_train = prepare_features(train_df, feature_cols, target_col)
        X_val, y_val = prepare_features(val_df, feature_cols, target_col)
        
        print(f"      Features prepared: {X_train.shape[1]} features")
        
        # Dictionary to store results
        results = {}
        
        
        # Tune Lasso Model
        print("\n[4/6] Tuning Lasso...")
        
        if models_config["lasso"]["enabled"]:
            lasso_param_grid = {
                "alpha": models_config["lasso"]["tuning"]["alpha_grid"]
            }
            
            print(f"\n  Grid Search Configuration:")
            print(f"    Alpha values: {lasso_param_grid['alpha']}")
            print(f"    CV folds: {training_config['cv_folds']}")
            print(f"    Total combinations: {len(lasso_param_grid['alpha'])}")
            
            _, _, val_metrics = tune_model_grid_search(
                base_model=Lasso(max_iter=1000),
                param_grid=lasso_param_grid,
                model_name="lasso",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                categorical_features=data_config["categorical_features"],
                numerical_features=data_config["numerical_features"],
                cv=training_config["cv_folds"]
            )
            results["lasso_tuned"] = val_metrics["rmse"]
        else:
            print("  ⊘ Lasso tuning disabled in config")
        
        
        # Tune Ridge model
        print("\n[5/6] Tuning Ridge...")
        
        if models_config["ridge"]["enabled"]:
            ridge_param_grid = {
                "alpha": models_config["ridge"]["tuning"]["alpha_grid"]
            }
            
            print(f"\n  Grid Search Configuration:")
            print(f"    Alpha values: {ridge_param_grid['alpha']}")
            print(f"    CV folds: {training_config['cv_folds']}")
            print(f"    Total combinations: {len(ridge_param_grid['alpha'])}")
            
            _, _, val_metrics = tune_model_grid_search(
                base_model=Ridge(),
                param_grid=ridge_param_grid,
                model_name="ridge",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                categorical_features=data_config["categorical_features"],
                numerical_features=data_config["numerical_features"],
                cv=training_config["cv_folds"]
            )
            results["ridge_tuned"] = val_metrics["rmse"]
        else:
            print("  ⊘ Ridge tuning disabled in config")
        
       

        # Tune Random Forest model
        print("\n[6/6] Tuning Random Forest...")
        
        if models_config["random_forest"]["enabled"]:
            rf_param_distributions = {
                "n_estimators": randint(100, 500),
                "max_depth": randint(10, 50),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10),
                "max_samples": uniform(0.7, 0.3)
            }
            
            n_iter = models_config["random_forest"]["tuning"]["n_iter"]
            
            print(f"\n  Random Search Configuration:")
            print(f"    n_estimators: randint(100, 500)")
            print(f"    max_depth: randint(10, 50)")
            print(f"    min_samples_split: randint(2, 20)")
            print(f"    min_samples_leaf: randint(1, 10)")
            print(f"    max_samples: uniform(0.7, 1.0)")
            print(f"    CV folds: {training_config['cv_folds']}")
            print(f"    Iterations: {n_iter}")
            
            _, _, val_metrics = tune_model_random_search(
                base_model=RandomForestRegressor(
                    random_state=training_config["random_state"],
                    n_jobs=training_config["n_jobs"]
                ),
                param_distributions=rf_param_distributions,
                model_name="random_forest",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                categorical_features=data_config["categorical_features"],
                numerical_features=data_config["numerical_features"],
                n_iter=n_iter,
                cv=training_config["cv_folds"]
            )
            results["random_forest_tuned"] = val_metrics["rmse"]
        else:
            print("  ⊘ Random Forest tuning disabled in config")
        
        
        # Compare results
        print("TUNING RESULTS SUMMARY")
        
        if not results:
            print("\n No models were tuned (all disabled in config)")
            return
        
        # Create results dataframe
        results_df = pd.DataFrame(
            list(results.items()),
            columns=["Model", "Validation RMSE"]
        )
        results_df = results_df.sort_values("Validation RMSE").reset_index(drop=True)
        
        # Display results
        print("\n")
        print(results_df.to_string(index=False))
        
        # Best tuned model
        best_tuned_model = results_df.iloc[0]["Model"]
        best_tuned_rmse = results_df.iloc[0]["Validation RMSE"]
        
        print(f"BEST TUNED MODEL: {best_tuned_model.upper()}")
        print(f"RMSE: {best_tuned_rmse:.2f} minutes")
        
        # Save results
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        results_file = models_dir / "tuning_comparison.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n✓ Results saved to: {results_file}")
        
    
        print(" HYPERPARAMETER TUNING COMPLETE!")
        print("\n View all results (baseline + tuned) in MLflow UI")
        print("\n Tuned models saved in: models/")
        print(f" Tuning results: {results_file}")
        print()
        
    except FileNotFoundError as e:
        print(f"\n FILE NOT FOUND ERROR: {e}")
        sys.exit(1)
        
    except ValueError as e:
        print(f"\n VALUE ERROR: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
