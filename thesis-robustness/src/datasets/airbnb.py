"""Airbnb Price Prediction dataset loader."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from ..common.registry import register_dataset


@register_dataset('airbnb')
def load_airbnb(data_dir: Path = None, **kwargs):
    """
    Load and preprocess Airbnb Price Prediction dataset.
    
    Returns:
        X: Feature matrix
        y: Target prices (log-transformed)
    """
    if data_dir is None:
        data_dir = Path('data/raw')
    
    # Look for downloaded Kaggle files
    # Kaggle typically downloads as a zip, so check for common filenames
    possible_files = [
        data_dir / 'airbnb.csv',
        data_dir / 'train.csv',
        data_dir / 'Airbnb_train.csv',  # Actual Kaggle competition filename
        data_dir / 'airbnb_price_prediction' / 'train.csv',
    ]
    
    data_path = None
    for path in possible_files:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(
            "Airbnb data not found. Please download using:\n"
            "  kaggle competitions download -c airbnb-price-prediction\n"
            "Then extract and place the CSV file in one of these locations:\n"
            f"  - {possible_files[0]}\n"
            f"  - {possible_files[1]}\n"
            f"  - {possible_files[2]}"
        )
    
    print(f"Loading Airbnb dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Basic preprocessing - adjust based on actual dataset structure
    # Common columns in Airbnb datasets: price, room_type, neighbourhood, etc.
    
    # Find price column (could be 'price', 'Price', 'log_price', etc.)
    price_cols = [col for col in df.columns if 'price' in col.lower()]
    if not price_cols:
        raise ValueError(
            f"No price column found. Available columns: {list(df.columns)}\n"
            "Please check the dataset structure."
        )
    
    price_col = price_cols[0]  # Use first matching column
    print(f"Using price column: {price_col}")
    
    # Drop rows with missing target
    df = df.dropna(subset=[price_col])
    y = df[price_col].values
    X = df.drop(price_col, axis=1)
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in categorical_cols:
        # Skip if too many unique values (likely IDs)
        if X[col].nunique() > 100:
            X = X.drop(col, axis=1)
            continue
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    # Fill missing values with median for numerical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
    
    # Convert to numpy (only numerical columns)
    X = X.select_dtypes(include=[np.number]).values
    
    # Log-transform target for regression (handles skewed distributions)
    y = np.log1p(y)  # log(1 + y) to handle zeros
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y
