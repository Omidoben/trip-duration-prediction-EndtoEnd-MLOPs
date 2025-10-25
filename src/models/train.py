import mlflow
import mlflow.sklearn
import pickle
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

from src.data.preprocessing import get_preprocessor

def setup_mlflow(tracking_uri, experiment_name):
    """
    Sets up the mlflow tracking
    """
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking uri: {mlflow.get_tracking_uri()}")

    experiment = mlflow.set_experiment(experiment_name)
    print(f"Experiment name: {experiment_name}")
    print(f"Experiment ID: {experiment.experiment_id}\n")

    return experiment.experiment_id


def evaluate_model(y_true, y_pred):
    """"
    Calculates model's evaluation metrics
    Returns:
        dict: Dictionary containing RMSE, MAE, R2 metrics
    """
    metrics = {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }
    return metrics



def train_model(model, model_name, X_train, y_train, X_val, y_val,
                categorical_features, numerical_features, save_path='models'):
    """
    This function:
    1. Creates a preprocessing + model pipeline
    2. Trains on training data
    3. Evaluates on both train and validation sets
    4. Logs everything to MLflow
    5. Saves the model locally
    """
    # Create preprocessing + model pipeline
    preprocessor = get_preprocessor(categorical_features, numerical_features)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    
    with mlflow.start_run(run_name=model_name):
        print(f"Training: {model_name}")
        
        pipeline.fit(X_train, y_train)
        
        # Predict on both train and validation sets
        y_pred_train = pipeline.predict(X_train)
        y_pred_val = pipeline.predict(X_val)
        
        # Evaluate
        train_metrics = evaluate_model(y_train, y_pred_train)
        val_metrics = evaluate_model(y_val, y_pred_val)
        
        # Log model parameters to MLflow
        mlflow.log_param("model_type", type(model).__name__)
        
        # Log all hyperparameters
        for param_name, param_value in model.get_params().items():
            mlflow.log_param(param_name, param_value)
        
        # Log all metrics
        for metric_name, metric_value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", metric_value)
        for metric_name, metric_value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", metric_value)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        
        # Print results
        print(f"\n  Train RMSE: {train_metrics['rmse']:.2f} minutes")
        print(f"  Val RMSE:   {val_metrics['rmse']:.2f} minutes")
        print(f"  Val MAE:    {val_metrics['mae']:.2f} minutes")
        print(f"  Val R²:     {val_metrics['r2']:.4f}")
        
        # Save model locally
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        model_file = save_dir / f"{model_name}.pkl"
        
        with open(model_file, "wb") as f:
            pickle.dump(pipeline, f)
        
        print(f"  Model saved: {model_file}")
    
    return pipeline, val_metrics



# Tune lasso and ridge models using Grid search CV
def tune_model_grid_search(base_model, param_grid, model_name,
                           X_train, y_train, X_val, y_val,
                           categorical_features, numerical_features,
                           cv=3, save_path="models"):
    """
    Tune model using GridSearchCV
    
    This function:
    1. Creates a pipeline with the model
    2. Uses GridSearchCV to test ALL combinations of parameters
    3. Logs the best results to MLflow
    4. Saves the best model
    """
    # Create pipeline
    preprocessor = get_preprocessor(categorical_features, numerical_features)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", base_model)
    ])
    
    # Adjust parameter names for pipeline (add 'regressor__' prefix)
    pipeline_param_grid = {
        f"regressor__{k}": v for k, v in param_grid.items()
    }
    
    with mlflow.start_run(run_name=f"{model_name}_tuned"):
        print(f"Tuning: {model_name} (GridSearchCV)")
        print(f"  Testing {sum(len(v) for v in param_grid.values())} parameter combinations...")
        print(f"  Cross-validation folds: {cv}\n")
        
        grid_search = GridSearchCV(
            pipeline,
            pipeline_param_grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_pipeline = grid_search.best_estimator_
        
        # Evaluate on validation set
        y_pred_val = best_pipeline.predict(X_val)
        val_metrics = evaluate_model(y_val, y_pred_val)
        
        # Log parameters to MLflow
        mlflow.log_param("model_type", type(base_model).__name__)
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("cv_folds", cv)
        mlflow.log_param("total_combinations_tested", 
                        sum(len(v) for v in param_grid.values()))
        
        # Log best parameters - cleaning up the names
        for param_name, param_value in grid_search.best_params_.items():
            clean_name = param_name.replace("regressor__", "")
            mlflow.log_param(f"best_{clean_name}", param_value)
        
        # Log metrics
        mlflow.log_metric("best_cv_rmse", -grid_search.best_score_)
        for metric_name, metric_value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", metric_value)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")
        
        # Print results
        print(f"\n  Best parameters found:")
        for param_name, param_value in grid_search.best_params_.items():
            clean_name = param_name.replace("regressor__", "")
            print(f"    - {clean_name}: {param_value}")
        
        print(f"\n  Best CV RMSE: {-grid_search.best_score_:.2f} minutes")
        print(f"  Val RMSE:     {val_metrics['rmse']:.2f} minutes")
        print(f"  Val MAE:      {val_metrics['mae']:.2f} minutes")
        print(f"  Val R²:       {val_metrics['r2']:.4f}")
        
        # Save model locally
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        model_file = save_dir / f"{model_name}_tuned.pkl"
        
        with open(model_file, "wb") as f:
            pickle.dump(best_pipeline, f)
        
        print(f"  Model saved: {model_file}")
    
    return best_pipeline, grid_search.best_params_, val_metrics



# Tune Random forest using Randomized Search CV
def tune_model_random_search(base_model, param_distributions, model_name,
                             X_train, y_train, X_val, y_val,
                             categorical_features, numerical_features,
                             n_iter=30, cv=3, save_path="models"):
    """
    Tune model using RandomizedSearchCV
    
    This function:
    1. Creates a pipeline with the model
    2. Uses RandomizedSearchCV to test RANDOM combinations of parameters
    3. Logs the best results to MLflow
    4. Saves the best model
    """
    # Create pipeline
    preprocessor = get_preprocessor(categorical_features, numerical_features)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", base_model)
    ])
    
    # Adjust parameter names for pipeline (add 'regressor__' prefix)
    pipeline_param_dist = {
        f"regressor__{k}": v for k, v in param_distributions.items()
    }
    
    with mlflow.start_run(run_name=f"{model_name}_tuned"):
        print(f"Tuning: {model_name} (RandomizedSearchCV)")
        print(f"  Testing {n_iter} random parameter combinations...")
        print(f"  Cross-validation folds: {cv}\n")
        
        random_search = RandomizedSearchCV(
            pipeline,
            pipeline_param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        # Get best model
        best_pipeline = random_search.best_estimator_
        
        # Evaluate on validation set
        y_pred_val = best_pipeline.predict(X_val)
        val_metrics = evaluate_model(y_val, y_pred_val)

        mlflow.set_tag("author", "Benard")
        
        # Log parameters to MLflow
        mlflow.log_param("model_type", type(base_model).__name__)
        mlflow.log_param("tuning_method", "RandomizedSearchCV")
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("cv_folds", cv)
       
        
        # Log best parameters (clean up the names)
        for param_name, param_value in random_search.best_params_.items():
            clean_name = param_name.replace("regressor__", "")
            mlflow.log_param(f"best_{clean_name}", param_value)
        
        # Log metrics
        mlflow.log_metric("best_cv_rmse", -random_search.best_score_)
        for metric_name, metric_value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", metric_value)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")
        
        # Print results
        print(f"\n  Best parameters found:")
        for param_name, param_value in random_search.best_params_.items():
            clean_name = param_name.replace("regressor__", "")
            print(f"    - {clean_name}: {param_value}")
        
        print(f"\n  Best CV RMSE: {-random_search.best_score_:.2f} minutes")
        print(f"  Val RMSE:     {val_metrics['rmse']:.2f} minutes")
        print(f"  Val MAE:      {val_metrics['mae']:.2f} minutes")
        print(f"  Val R²:       {val_metrics['r2']:.4f}")
        
        # Save model locally
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        model_file = save_dir / f"{model_name}_tuned.pkl"
        
        with open(model_file, "wb") as f:
            pickle.dump(best_pipeline, f)
        
        print(f"  Model saved: {model_file}")
    
    return best_pipeline, random_search.best_params_, val_metrics
