import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.load_data import load_train_val_data
from src.data.preprocessing import prepare_features, get_preprocessor
from src.utils.config_loader import load_config
import numpy as np


def test_preprocessing():
    try:
        config = load_config("configs/config.yaml")
        data_config = config["data"]
        
        print(f"\nLoading and preparing data...")
        train_df, val_df = load_train_val_data(config)
        
        feature_cols = (
            data_config["categorical_features"] + 
            data_config["numerical_features"]
        )
        target_col = data_config["target"]
        
        X_train, y_train = prepare_features(train_df, feature_cols, target_col)
        X_val, y_val = prepare_features(val_df, feature_cols, target_col)
        print(f"Data prepared")
        
        # Create preprocessor
        print(f"\nCreating preprocessor...")
        preprocessor = get_preprocessor(
            data_config["categorical_features"],
            data_config["numerical_features"]
        )
        print(f"Preprocessor created")
        
        # Fit and transform
        print(f"\nFitting preprocessor on training data...")
        X_train_transformed = preprocessor.fit_transform(X_train)
        print(f"Fit successful")
        
        # Transform validation
        print(f"Transforming validation data...")
        X_val_transformed = preprocessor.transform(X_val)
        print(f"Transform successful")
        
        # Display results
        print(f"\nTransformed Data:")
        print(f"  X_train_transformed shape: {X_train_transformed.shape}")
        print(f"  X_train_transformed type: {type(X_train_transformed)}")
        print(f"  X_val_transformed shape: {X_val_transformed.shape}")
        
        print("\nPreprocessing test passed")
        return True
        
    except Exception as e:
        print(f"\nPreprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_preprocessing()
    sys.exit(0 if success else 1)