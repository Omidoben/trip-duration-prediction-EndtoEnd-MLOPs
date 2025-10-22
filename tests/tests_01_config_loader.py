import sys
from pathlib import Path

# add project root to path
project_root =  Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config

def test_config_loader():
    """Test configuration loading works as expected"""
    try:
        config = load_config("configs/config.yaml")
        print("Config loaded successfully")

        # verify all sections exist
        assert "data" in config, "Missing 'data' section"
        assert "mlflow" in config, "Missing 'mlflow' section"
        assert "training" in config, "Missing 'training' section"
        assert "models" in config, "Missing 'models' section"

        # display key values
        print("\n Config summary: ")
        print(f"Experiment: {config['mlflow']['experiment_name']}")
        print(f"Train file: {config['data']['train_file']}")
        print(f"Val file: {config['data']['val_file']}")
        print(f"Categorical features: {config['data']['categorical_features']}")
        print(f"Numerical features: {config['data']['numerical_features']}")
        print(f"Target: {config['data']['target']}")

        print("\nConfig test passed")
        return True
    except Exception as e:
        print(f"Config test failed: {e}")
        import traceback
        traceback.print_exc
        return False
    

if __name__=="__main__":
    success = test_config_loader()
    sys.exit(0 if success else 1)