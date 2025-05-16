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
        scaler = load(SCALER_MODEL)  # Load the scaler
        logging.info("Models loaded successfully.")
        return iso_model, knn_imputer, scaler  # Return the scaler
    except FileNotFoundError:
        logging.error("Model file not found. Ensure models have been trained.")
        return None, None, None
    except Exception as e:
        logging.error(f"Error loading models: {e}", exc_info=True)
        return None, None, None

def analyze_dataset(df):
    """
    Analyzes dataset and returns a dictionary of AI-powered cleaning suggestions.
    """
    suggestions = {}
    num_cols = df.select_dtypes(include=np.number).columns

    # Load models
    iso_model, knn_imputer, scaler = load_models()  # Load the scaler
    if iso_model is None or knn_imputer is None or scaler is None:
        return suggestions  # Return empty suggestions if models failed to load

    # Predict outliers
    if not num_cols.empty:
        # Impute missing values before outlier detection
        imputer = SimpleImputer(strategy=IMPUTER_STRATEGY)
        X = df[num_cols].copy()
        X[num_cols] = imputer.fit_transform(X[num_cols])

        # Scale the data for Isolation Forest
        X[num_cols] = scaler.transform(X[num_cols])  # Use transform, not fit_transform

        preds = iso_model.predict(X)  # -1 = outlier
        outlier_count = (preds == -1).sum()
        if outlier_count > 0:
            suggestions["Outliers"] = f"{outlier_count} rows may be outliers (Isolation Forest)."

    # Missing values
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        if df[col].dtype in ["float64", "int64"]:
            suggestion = f"Impute missing values in '{col}' using KNN imputer (n_neighbors={KNN_NEIGHBORS})"
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == "object":
            suggestion = f"For column '{col}', consider filling missing values with the mode (most frequent value)"
        else:
            suggestion = f"Consider how to best fill missing values in '{col}'" # More general case
        suggestions[col] = {
            "missing": f"{missing[col]} missing",
            "recommendation": suggestion
        }

    return suggestions