import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import logging  # Added logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DIR = "models"
ISOLATION_MODEL = os.path.join(MODEL_DIR, "isolation_forest.pkl")
IMPUTER_MODEL = os.path.join(MODEL_DIR, "knn_imputer.pkl")
SCALER_MODEL = os.path.join(MODEL_DIR, "scaler.pkl")  # Added scaler model path
IMPUTER_STRATEGY = "mean"  # Or "median"
ISO_CONTAMINATION = 0.05
KNN_NEIGHBORS = 5

os.makedirs(MODEL_DIR, exist_ok=True)

def train_models(train_df):
    """
    Trains Isolation Forest and KNN Imputer on a training dataset.
    """
    try:
        num_cols = train_df.select_dtypes(include=np.number).columns
        if num_cols.empty:
            logging.warning("No numerical columns found for training models.")
            return

        # Impute missing values *before* training Isolation Forest
        imputer = SimpleImputer(strategy=IMPUTER_STRATEGY)
        X = train_df[num_cols].copy()  # Work on a copy to avoid modifying original DataFrame
        X[num_cols] = imputer.fit_transform(X[num_cols])

        # Scale the data and fit the scaler
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])  # FIT AND TRANSFORM
        dump(scaler, SCALER_MODEL)  # Save the fitted scaler
        logging.info(f"Scaler model trained and saved to {SCALER_MODEL}")

        # Train Isolation Forest
        iso_model = IsolationForest(contamination=ISO_CONTAMINATION, random_state=42)
        iso_model.fit(X)
        dump(iso_model, ISOLATION_MODEL)
        logging.info(f"Isolation Forest model trained and saved to {ISOLATION_MODEL}")

        # Train KNN Imputer (train on unscaled data)
        knn_imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
        knn_imputer.fit(train_df[num_cols])  # Train on original, potentially missing data
        dump(knn_imputer, IMPUTER_MODEL)
        logging.info(f"KNN Imputer model trained and saved to {IMPUTER_MODEL}")

        logging.info("✅ Models trained and saved.")

    except Exception as e:
        logging.error(f"Error training models: {e}", exc_info=True)  # Log with traceback

def load_models():
    """Loads trained ML models."""
    try:
        iso_model = load(ISOLATION_MODEL)
        knn_imputer = load(IMPUTER_MODEL)
        scaler = load(SCALER_MODEL)
        logging.info("Models loaded successfully.")
        return iso_model, knn_imputer, scaler
    except FileNotFoundError:
        logging.error("Model file not found. Ensure models have been trained.")
        # Ensure we always return 3 values, even if None
        return None, None, None
    except Exception as e:
        logging.error(f"Error loading models: {e}", exc_info=True)
        # Ensure we always return 3 values, even if None
        return None, None, None

def handle_missing_values(df, strategy='auto', custom_value=None):
    """
    Handles missing values using various strategies.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): One of ['auto', 'knn', 'mean', 'median', 'mode', 'custom', 'drop']
        custom_value: Value to use when strategy is 'custom'
    
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    try:
        df_copy = df.copy()
        num_cols = df_copy.select_dtypes(include=np.number).columns
        cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns
        
        if strategy == 'drop':
            return df_copy.dropna()
            
        elif strategy == 'auto':
            # Use KNN for numerical columns and mode for categorical
            if not num_cols.empty:
                knn_imputer = load(IMPUTER_MODEL)
                if knn_imputer is not None:
                    df_copy[num_cols] = knn_imputer.transform(df_copy[num_cols])
                else:
                    # Fallback to mean if KNN model not available
                    df_copy[num_cols] = SimpleImputer(strategy='mean').fit_transform(df_copy[num_cols])
            
            # Use mode for categorical columns
            for col in cat_cols:
                mode_value = df_copy[col].mode()[0] if not df_copy[col].mode().empty else None
                if mode_value is not None:
                    df_copy[col] = df_copy[col].fillna(mode_value)
                    
        elif strategy == 'knn':
            if not num_cols.empty:
                knn_imputer = load(IMPUTER_MODEL)
                if knn_imputer is not None:
                    df_copy[num_cols] = knn_imputer.transform(df_copy[num_cols])
                else:
                    logging.warning("KNN imputer model not found. Falling back to mean imputation.")
                    df_copy[num_cols] = SimpleImputer(strategy='mean').fit_transform(df_copy[num_cols])
                    
        elif strategy in ['mean', 'median']:
            if not num_cols.empty:
                df_copy[num_cols] = SimpleImputer(strategy=strategy).fit_transform(df_copy[num_cols])
                
        elif strategy == 'mode':
            for col in df_copy.columns:
                mode_value = df_copy[col].mode()[0] if not df_copy[col].mode().empty else None
                if mode_value is not None:
                    df_copy[col] = df_copy[col].fillna(mode_value)
                    
        elif strategy == 'custom' and custom_value is not None:
            df_copy = df_copy.fillna(custom_value)
            
        return df_copy
        
    except Exception as e:
        logging.error(f"Error in handle_missing_values: {e}", exc_info=True)
        return df

def handle_outliers(df, strategy='remove', custom_value=None):
    """
    Handles outliers using various strategies based on Isolation Forest predictions.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): One of ['remove', 'replace_boundary', 'custom']
        custom_value: Value to use when strategy is 'custom'
    
    Returns:
        pd.DataFrame: DataFrame with outliers handled
    """
    try:
        print("Entering handle_outliers function") # Added print statement
        df_copy = df.copy()
        num_cols = df_copy.select_dtypes(include=np.number).columns
        
        if num_cols.empty:
            logging.warning("No numerical columns found for outlier handling.")
            return df

        logging.info("Calling load_models() in handle_outliers...")
        loaded_models = load_models()
        logging.info(f"Returned from load_models(): Type={type(loaded_models).__name__}, Length={len(loaded_models) if isinstance(loaded_models, (list, tuple)) else 'N/A'}")

        # Defensive check for unpacking - expect 3 items
        if not (isinstance(loaded_models, tuple) and len(loaded_models) == 3):
             logging.error(f"Expected a tuple of 3 models from load_models(), but got {type(loaded_models).__name__} with length {len(loaded_models) if isinstance(loaded_models, (list, tuple)) else 'N/A'}.")
             return df # Return original df on unexpected load failure

        logging.info("Attempting to unpack loaded_models...")
        iso_model, knn_imputer, scaler = loaded_models # Unpack all 3
        logging.info("Successfully unpacked models.")

        # Now check if the specific models needed are None
        if iso_model is None or scaler is None:
            logging.error("Isolation Forest model or scaler is None after loading.")
            return df

        # Need to impute numerical data before scaling for outlier prediction
        imputer = SimpleImputer(strategy=IMPUTER_STRATEGY) # Use the default imputer strategy
        X = df_copy[num_cols].copy()
        X[num_cols] = imputer.fit_transform(X[num_cols])

        X[num_cols] = scaler.transform(X[num_cols]) # Scale the data
        
        # Get outlier predictions (-1 for outliers, 1 for inliers)
        outlier_preds = iso_model.predict(X)
        outlier_mask = (outlier_preds == -1)

        if strategy == 'remove':
            # Remove rows where Isolation Forest predicted an outlier
            df_cleaned = df_copy[~outlier_mask].reset_index(drop=True)
            logging.info(f"Removed {outlier_mask.sum()} outliers.")
            return df_cleaned
            
        elif strategy == 'replace_boundary':
            # Replace outliers with boundary values (using IQR for simplicity here, 
            # as Isolation Forest doesn't directly provide boundaries)
            # Note: A more sophisticated approach might involve understanding the distribution 
            # learned by Isolation Forest, but IQR is a common practical approach for boundary replacement.
            for col in num_cols:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Replace values below lower bound with lower bound
                df_copy.loc[df_copy[col] < lower_bound, col] = lower_bound
                # Replace values above upper bound with upper bound
                df_copy.loc[df_copy[col] > upper_bound, col] = upper_bound
                
            logging.info(f"Replaced outliers with boundary values.")
            return df_copy
            
        elif strategy == 'custom' and custom_value is not None:
            # Replace outliers with a custom value
            for col in num_cols:
                df_copy.loc[outlier_mask, col] = custom_value
            logging.info(f"Replaced outliers with custom value: {custom_value}.")
            return df_copy
            
        else:
            logging.warning(f"Unknown outlier handling strategy: {strategy}. Returning original DataFrame.")
            return df_copy
            
    except Exception as e:
        logging.error(f"Error in handle_outliers: {e}", exc_info=True)
        return df

def analyze_dataset(df):
    """
    Analyzes dataset and returns a dictionary of AI-powered cleaning suggestions.
    """
    suggestions = {}
    num_cols = df.select_dtypes(include=np.number).columns

    # Load models
    logging.info("Calling load_models() in analyze_dataset...") # Added logging
    loaded_models = load_models()
    logging.info(f"Returned from load_models() in analyze_dataset: Type={type(loaded_models).__name__}, Length={len(loaded_models) if isinstance(loaded_models, (list, tuple)) else 'N/A'}") # Added logging

    # Defensive check for unpacking - expect 3 items
    if not (isinstance(loaded_models, tuple) and len(loaded_models) == 3):
        logging.error(f"Expected a tuple of 3 models from load_models() in analyze_dataset, but got {type(loaded_models).__name__} with length {len(loaded_models) if isinstance(loaded_models, (list, tuple)) else 'N/A'}.") # Modified logging
        return suggestions # Return empty suggestions on unexpected load failure

    logging.info("Attempting to unpack loaded_models in analyze_dataset...") # Added logging
    iso_model, knn_imputer, scaler = loaded_models # Unpack all 3
    logging.info("Successfully unpacked models in analyze_dataset.") # Added logging

    # Now check if any of the specific models needed are None
    if iso_model is None or knn_imputer is None or scaler is None:
        logging.warning("One or more models are None after loading in analyze_dataset.") # Added logging
        return suggestions

    # Predict outliers
    if not num_cols.empty:
        # Impute missing values before outlier detection
        imputer = SimpleImputer(strategy=IMPUTER_STRATEGY)
        X = df[num_cols].copy()
        X[num_cols] = imputer.fit_transform(X[num_cols])

        # Scale the data for Isolation Forest
        X[num_cols] = scaler.transform(X[num_cols])

        preds = iso_model.predict(X)
        outlier_count = (preds == -1).sum()
        if outlier_count > 0:
            suggestions["Outliers"] = {
                "count": outlier_count,
                "message": f"{outlier_count} rows may be outliers (Isolation Forest).",
                "recommendation": "Consider removing or investigating these outliers."
            }

    # Missing values analysis
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    for col in missing[missing > 0].index:
        col_type = df[col].dtype
        missing_count = missing[col]
        missing_pct = missing_percent[col]
        
        if col_type in ["float64", "int64"]:
            if missing_pct < 5:  # Less than 5% missing
                suggestion = f"Fill missing values in '{col}' with mean (only {missing_pct:.1f}% missing)"
            elif missing_pct < 30:  # Less than 30% missing
                suggestion = f"Use KNN imputer for '{col}' ({missing_pct:.1f}% missing)"
            else:  # More than 30% missing
                suggestion = f"Consider dropping column '{col}' or using advanced imputation ({missing_pct:.1f}% missing)"
        elif pd.api.types.is_categorical_dtype(col_type) or col_type == "object":
            suggestion = f"Fill missing values in '{col}' with mode (most frequent value)"
        else:
            suggestion = f"Investigate missing values in '{col}' ({missing_pct:.1f}% missing)"
            
        suggestions[col] = {
            "missing_count": int(missing_count),
            "missing_percent": float(missing_pct),
            "recommendation": suggestion,
            "type": str(col_type)
        }

    return suggestions