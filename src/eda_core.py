import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_summary_statistics(df):
    """Returns descriptive statistics."""
    try:
        return df.describe(include='all')
    except Exception as e:
        print(f"Error in get_summary_statistics: {e}")
        return None  # Or raise the exception if you want the app to halt


def get_basic_info(df):
    """Returns dataset shape, column names, and data types."""
    try:
        return {
            "Shape": df.shape,
            "Columns": list(df.columns),
            "Data Types": df.dtypes.astype(str).to_dict()
        }
    except Exception as e:
        print(f"Error in get_basic_info: {e}")
        return None


def get_missing_values(df):
    """Returns a DataFrame with count and percentage of missing values."""
    try:
        missing = df.isnull().sum()
        percent = (missing / len(df)) * 100
        summary = pd.DataFrame({
            'Missing Values': missing,
            'Percentage (%)': percent
        })
        return summary[summary['Missing Values'] > 0].sort_values(by='Percentage (%)', ascending=False)
    except Exception as e:
        print(f"Error in get_missing_values: {e}")
        return pd.DataFrame()  # Return an empty DataFrame


def clean_missing_values(df, method='drop', fill_value=None):
    """Cleans missing data using selected method."""
    try:
        if method == 'drop':
            return df.dropna()
        elif method == 'fill':
            return df.fillna(fill_value)
        return df
    except Exception as e:
        print(f"Error in clean_missing_values: {e}")
        return df  # Return the original DataFrame


def get_duplicate_info(df):
    """Returns count of duplicate rows."""
    try:
        return df.duplicated().sum()
    except Exception as e:
        print(f"Error in get_duplicate_info: {e}")
        return 0


def detect_outliers(df, output_dir):
    """Detects outliers using IQR and saves boxplots."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        outliers = {}
        for col in df.select_dtypes(include=np.number).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask = (df[col] < lower) | (df[col] > upper)
            if mask.sum() > 0:
                outliers[col] = int(mask.sum())
                save_boxplot(df, col, output_dir)
        return outliers
    except Exception as e:
        print(f"Error in detect_outliers: {e}")
        return {}


def save_boxplot(df, col, output_dir):
    """Saves a boxplot for a column."""
    try:
        plt.figure(figsize=(6, 2))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.savefig(os.path.join(output_dir, f"boxplot_{col}.png"))
        plt.close()
    except Exception as e:
        print(f"Error in save_boxplot: {e}")