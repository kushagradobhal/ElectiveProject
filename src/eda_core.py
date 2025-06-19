import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_summary_statistics(df):
    """Returns descriptive statistics and data quality profiling."""
    try:
        summary = pd.DataFrame(index=df.columns)
        summary['Data Type'] = df.dtypes.astype(str)
        summary['Classified Type'] = classify_column_types(df).values()
        summary['Non-Null Count'] = df.notnull().sum()
        summary['Null Count'] = df.isnull().sum()
        summary['Null Percentage (%)'] = (df.isnull().sum() / len(df)) * 100
        summary['Unique Count'] = df.nunique()
        summary['Duplicate Count'] = df.duplicated().sum() # Row-wise duplicates, per column doesn't make sense here

        # Add descriptive statistics based on type
        for col in df.columns:
            if summary.loc[col, 'Classified Type'] == 'Numerical':
                summary.loc[col, 'Mean'] = df[col].mean()
                summary.loc[col, 'Median'] = df[col].median()
                summary.loc[col, 'Standard Deviation'] = df[col].std()
                summary.loc[col, 'Min'] = df[col].min()
                summary.loc[col, 'Max'] = df[col].max()
                summary.loc[col, 'Skewness'] = df[col].skew()
                summary.loc[col, 'Kurtosis'] = df[col].kurtosis()
            elif summary.loc[col, 'Classified Type'] == 'Categorical' or summary.loc[col, 'Classified Type'] == 'Text':
                 mode_result = df[col].mode()
                 if not mode_result.empty:
                     summary.loc[col, 'Most Frequent Value'] = mode_result.iloc[0]
                     summary.loc[col, 'Frequency'] = df[col].value_counts().iloc[0]

        return summary
    except Exception as e:
        print(f"Error in get_summary_statistics: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


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


def classify_column_types(df):
    """
    Classifies columns into Numerical, Categorical, Text, and Datetime.
    """
    column_types = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            column_types[col] = 'Numerical'
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            column_types[col] = 'Datetime'
        elif pd.api.types.is_categorical_dtype(dtype) or dtype == 'object':
            # Further classify object types as Categorical or Text
            # This is a basic approach; more sophisticated text detection might be needed later
            if df[col].nunique() < 50 and len(df) / df[col].nunique() > 10: # Heuristic for categorical
                column_types[col] = 'Categorical'
            else:
                column_types[col] = 'Text' # Default to Text if not clearly categorical
        else:
            column_types[col] = str(dtype) # Fallback to pandas dtype string
    return column_types


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


def detect_outliers_zscore(df, threshold=3):
    """
    Detects outliers using Z-Score.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Z-score threshold to consider a value an outlier.
        
    Returns:
        dict: A dictionary with column names and the count of outliers detected.
    """
    outliers = {}
    num_cols = df.select_dtypes(include=np.number).columns
    
    if num_cols.empty:
        return outliers
        
    for col in num_cols:
        # Calculate Z-scores, ignoring NaNs
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        # Count outliers based on threshold, ignoring NaNs in the original column
        outlier_mask = (z_scores > threshold) & (df[col].notna())
        count = outlier_mask.sum()
        if count > 0:
            outliers[col] = int(count)
            
    return outliers


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


def generate_density_summary(df):
    """
    Generates natural-language summaries of value density for numerical columns.
    """
    density_summaries = {}
    num_cols = df.select_dtypes(include=np.number).columns

    if num_cols.empty:
        return density_summaries

    for col in num_cols:
        # Calculate 25th and 75th percentiles
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        # Handle cases with insufficient data or constant values
        if pd.isna(Q1) or pd.isna(Q3) or Q1 == Q3:
            summary = f"Distribution for '{col}' could not be summarized." # Or provide a different summary
        else:
            summary = f"Approximately 50% of values in '{col}' are concentrated between {Q1:.2f} and {Q3:.2f}."

        density_summaries[col] = summary

    return density_summaries

def save_histogram(df, col, output_dir):
    """
    Saves a histogram for a numerical column.
    """
    try:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f"histogram_{col}.png"))
        plt.close()
    except Exception as e:
        print(f"Error in save_histogram for {col}: {e}")

def save_bar_chart(df, col, output_dir, top_n=10):
    """
    Saves a bar chart for a categorical column (shows top_n value counts).
    """
    try:
        plt.figure(figsize=(8, 4))
        # Get top N values, handle potential NaNs
        top_values = df[col].value_counts().nlargest(top_n).index.tolist()
        # Filter dataframe to include only top values for consistent plotting
        df_filtered = df[df[col].isin(top_values)]
        
        sns.countplot(data=df_filtered, y=col, order=top_values)
        plt.title(f'Top {top_n} Categories in {col}')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        plt.savefig(os.path.join(output_dir, f"bar_chart_{col}.png"))
        plt.close()
    except Exception as e:
        print(f"Error in save_bar_chart for {col}: {e}")

def save_correlation_heatmap(df, output_dir):
    """
    Saves a correlation heatmap for numerical columns.
    """
    os.makedirs(output_dir, exist_ok=True)
    corr = df.select_dtypes(include='number').corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()

def save_pairplot(df, output_dir):
    """
    Saves a pairplot for numerical columns.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols) > 1:
        plt.figure()
        pairplot = sns.pairplot(df[num_cols].dropna())
        pairplot.fig.suptitle('Pairplot of Numerical Features', y=1.02)
        pairplot.savefig(os.path.join(output_dir, "pairplot.png"))
        plt.close()
