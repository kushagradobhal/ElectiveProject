import pandas as pd
from . import eda_core
from . import auto_analyzer
import datetime
import os
from weasyprint import HTML
import glob

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

def build_html_report(report_data, transformations_log, before_stats=None, after_stats=None, output_path="reports/report.html"):
    """
    Builds a comprehensive HTML report from the EDA data, cleaning log, and before/after stats.
    Automatically includes all visualization images from the reports/ directory and subdirectories.
    """
    html = []
    html.append(f"<html><head><title>AutoEDA Report</title></head><body>")
    html.append(f"<h1>AutoEDA EDA & Cleaning Report</h1>")
    html.append(f"<h2>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h2>")

    # Dataset Overview
    html.append("<h2>Dataset Overview</h2>")
    basic_info = report_data.get('basic_info', {})
    html.append("<ul>")
    for k, v in basic_info.items():
        html.append(f"<li><b>{k}:</b> {v}</li>")
    html.append("</ul>")

    # Column Types
    html.append("<h3>Column Types</h3>")
    col_types = report_data.get('column_types', {})
    html.append("<ul>")
    for k, v in col_types.items():
        html.append(f"<li><b>{k}:</b> {v}</li>")
    html.append("</ul>")

    # Summary Statistics (Before/After)
    html.append("<h2>Summary Statistics</h2>")
    summary_stats = report_data.get('summary_statistics', {})
    if summary_stats and isinstance(summary_stats, dict) and any(isinstance(v, dict) for v in summary_stats.values()):
        html.append(pd.DataFrame.from_dict(summary_stats).to_html())
    else:
        html.append("<i>No summary statistics available.</i>")
    if before_stats is not None:
        html.append("<h4>Before Cleaning</h4>")
        html.append(before_stats.to_html())
    if after_stats is not None:
        html.append("<h4>After Cleaning (from log)</h4>")
        html.append(after_stats.to_html())

    # Missing Values
    html.append("<h2>Missing Values Analysis</h2>")
    missing = report_data.get('missing_values', {})
    if missing and isinstance(missing, dict) and any(isinstance(v, dict) for v in missing.values()):
        html.append(pd.DataFrame.from_dict(missing).to_html())
    else:
        html.append("<i>No missing values data available.</i>")

    # Outlier Detection
    html.append("<h2>Outlier Detection</h2>")
    for method in ['outliers_isolation_forest', 'outliers_iqr', 'outliers_zscore']:
        outliers = report_data.get(method, {})
        html.append(f"<h4>{method.replace('_', ' ').title()}</h4>")
        if outliers:
            if isinstance(outliers, dict):
                html.append(pd.DataFrame.from_dict(outliers, orient='index').to_html())
            else:
                html.append(f"<pre>{outliers}</pre>")
        else:
            html.append("<i>No outliers detected or not applicable.</i>")

    # Density Summaries
    html.append("<h2>Density Summaries</h2>")
    density = report_data.get('density_summaries', {})
    if density and isinstance(density, dict) and any(isinstance(v, dict) for v in density.values()):
        html.append(pd.DataFrame.from_dict(density).to_html())
    elif density and isinstance(density, dict):
        # If it's a dict of strings, show as a list
        html.append("<ul>")
        for k, v in density.items():
            html.append(f"<li><b>{k}:</b> {v}</li>")
        html.append("</ul>")
    else:
        html.append("<i>No density summaries available.</i>")

    # Cleaning Actions Log
    html.append("<h2>Cleaning Actions Log</h2>")
    if transformations_log:
        html.append("<table border='1'><tr><th>Timestamp</th><th>Action Type</th><th>Columns</th><th>Parameters</th></tr>")
        for entry in transformations_log:
            html.append(f"<tr><td>{entry.get('timestamp','')}</td><td>{entry.get('action_type','')}</td><td>{entry.get('columns','')}</td><td>{entry.get('parameters','')}</td></tr>")
        html.append("</table>")
    else:
        html.append("<i>No cleaning actions logged.</i>")

    # Visualizations (auto-collected)
    html.append("<h2>Visualizations</h2>")
    # Boxplots
    boxplots = sorted(glob.glob("reports/boxplots/boxplot_*.png"))
    if boxplots:
        html.append("<h3>Boxplots</h3>")
        for img in boxplots:
            html.append(f"<img src='{img}' width='400'><br>")
    # Histograms
    histograms = sorted(glob.glob("reports/visualizations/histogram_*.png"))
    if histograms:
        html.append("<h3>Histograms</h3>")
        for img in histograms:
            html.append(f"<img src='{img}' width='400'><br>")
    # Bar Charts
    bar_charts = sorted(glob.glob("reports/visualizations/bar_chart_*.png"))
    if bar_charts:
        html.append("<h3>Bar Charts</h3>")
        for img in bar_charts:
            html.append(f"<img src='{img}' width='400'><br>")
    # Missing Value Visualizations
    missingno_imgs = ["reports/missingno_matrix.png", "reports/missingno_bar.png"]
    missingno_imgs = [img for img in missingno_imgs if os.path.exists(img)]
    if missingno_imgs:
        html.append("<h3>Missing Value Visualizations</h3>")
        for img in missingno_imgs:
            html.append(f"<img src='{img}' width='400'><br>")
    # Heatmaps
    heatmaps = sorted(glob.glob("reports/heatmaps/*.png"))
    if heatmaps:
        html.append("<h3>Heatmaps</h3>")
        for img in heatmaps:
            html.append(f"<img src='{img}' width='400'><br>")
    html.append("</body></html>")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    return output_path

def build_pdf_report(html_path, pdf_path="reports/report.pdf"):
    base_url = os.path.dirname(os.path.abspath(html_path))
    HTML(html_path, base_url=base_url).write_pdf(pdf_path)
    return pdf_path
