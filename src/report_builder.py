import pandas as pd
from . import eda_core
from . import auto_analyzer

def generate_eda_report_data(df):
    """
    Generates a dictionary containing all EDA report data and metadata.
    """
    report_data = {}

    # Basic Information and Column Types
    report_data['basic_info'] = eda_core.get_basic_info(df)
    report_data['column_types'] = eda_core.classify_column_types(df)

    # Summary Statistics and Data Quality
    report_data['summary_statistics'] = eda_core.get_summary_statistics(df).to_dict(orient='index')

    # Missing Values Analysis
    missing_values_summary_df = eda_core.get_missing_values(df)
    report_data['missing_values'] = missing_values_summary_df.to_dict(orient='index')
    # Note: Visualizations (missingno plots) are handled in the UI display function as they are images

    # Outlier Detection
    # Get Isolation Forest results (also ensures models are trained/loaded)
    auto_analyzer_suggestions = auto_analyzer.analyze_dataset(df)
    report_data['outliers_isolation_forest'] = auto_analyzer_suggestions.get('Outliers', None)

    # Get IQR and Z-Score results
    report_data['outliers_iqr'] = eda_core.detect_outliers(df, output_dir=None) # Don't save plots here
    report_data['outliers_zscore'] = eda_core.detect_outliers_zscore(df)
    # Note: Outlier visualizations (boxplots) are handled in the UI display function

    # Density Summary
    report_data['density_summaries'] = eda_core.generate_density_summary(df)

    # Note: Other visualizations (histograms, bar charts, heatmap, pairplot) are handled in the UI display function

    return report_data

# This function could be added later for generating a full HTML report if needed
# def build_html_report(report_data):
#     """
#     Builds a comprehensive HTML report from the EDA data.
#     """
#     pass
