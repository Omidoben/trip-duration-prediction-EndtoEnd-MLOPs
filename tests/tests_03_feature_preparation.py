import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.load_data import load_train_val_data
from src.data.preprocessing import prepare_features
from src.utils.config_loader import load_config


def test_feature_preparation():
    """Test feature preparation and splitting"""
    try:
        config = load_config("configs/config.yaml")
        data_config = config["data"]
        
        print(f"\nLoading data...")
        train_df, val_df = load_train_val_data(config)
        
        # Prepare features
        feature_cols = (
            data_config["categorical_features"] + 
            data_config["numerical_features"]
        )
        target_col = data_config["target"]
        
        print(f"Preparing features...")
        X_train, y_train = prepare_features(train_df, feature_cols, target_col)
        X_val, y_val = prepare_features(val_df, feature_cols, target_col)
        print(f"Features prepared")
        
        # Validate shapes
        print(f"\nShapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_val: {X_val.shape}")
        print(f"  y_val: {y_val.shape}")
        
        # Validate shapes match
        assert X_train.shape[0] == y_train.shape[0], "Train shapes don't match"
        assert X_val.shape[0] == y_val.shape[0], "Val shapes don't match"
        print(f"Shapes validated")
        
        # Validate columns
        print(f"\nFeature Columns ({len(feature_cols)}):")
        print(f"  Categorical: {data_config['categorical_features']}")
        print(f"  Numerical: {data_config['numerical_features']}")
        assert list(X_train.columns) == feature_cols, "Column mismatch"
        print(f"All features present")
        
        print("\nFeature preparation test passed")
        return True
        
    except Exception as e:
        print(f"\nFeature preparation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_feature_preparation()
    sys.exit(0 if success else 1)