"""Configuration loader utility"""
import yaml
from pathlib import Path


def load_config(config_path="configs/config.yaml"):
    """
    Load configutation from YAML file 

    Args:
        config_path: Path to config to file
    
    Returns:
        dict: Configuration dictionary
    """
    with open (config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_data_config(config):
    """Extract data configuration"""
    return config["data"]


def get_mlflow_config(config):
    """Extract MLflow configuration"""
    return config["mlflow"]


def get_training_config(config):
    """Extract training configuration"""
    return config["training"]


def get_model_config(config, model_name):
    """
    Extract specific model configuration
    
    Args:
        config: Full config dict
        model_name: Name of the model (e.g. Lasso)

    Returns:
        dict: Model configuration
    """
    return config["models"].get(model_name, {})