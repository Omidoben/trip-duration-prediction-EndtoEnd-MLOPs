import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.load_data import load_train_val_data
from src.utils.config_loader import load_config


def test_data_loading():
    """Tests data loading and validation"""
    try:
        config = load_config("configs/config.yaml")
        data_config = config["data"]

        # verify data files exist
        train_file = Path(data_config["raw_dir"]) / data_config["train_file"]
        val_file = Path(data_config["raw_dir"]) / data_config["val_file"]

        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")
        print(f"Train file exists: {train_file}")
        
        if not val_file.exists():
            raise FileNotFoundError(f"Validation file not found: {val_file}")
        print(f"Validation file exists: {val_file}")

        print("Loading data...")
        train_df, val_df = load_train_val_data(config)
        print("Data loaded successfully")

        # Validate shapes
        print("\nData Shapes:")
        print(f" Train: {train_df.shape}")
        print(f" Val: {val_df.shape}")

        # Validate columns
        print(f"\nValidating required columns...")
        required_cols = ["duration", "trip_distance", "passenger_count", "PU_DO"]
        for col in required_cols:
            assert col in train_df.columns, f"Missing column in train: {col}"
            assert col in val_df.columns, f"Missing column in val: {col}"
        print(f"All required columns present")
        

        print("\n Data loading test passed")
        return True
    except Exception as e:
        print(f"Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
if __name__=="__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)