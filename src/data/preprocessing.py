"""Data preprocessing and feature engineering"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def get_preprocessor(categorical_features, numerical_features):
    """
    Create preprocessing pipeline

    """
    # Categorical transformer
    cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    # Numerical transformer
    num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer([
    ("cat", cat_transformer, categorical_features),
    ("num", num_transformer, numerical_features)
    ])
    
    return preprocessor


def prepare_features(df, feature_cols, target_col):
    """
    Split dataframe into features and target

    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    return X, y