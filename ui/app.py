import streamlit as st
import pandas as pd
import os
import sys
import numpy as np
import missingno as msno 
import matplotlib.pyplot as plt
import datetime
import glob

# Initialize the transformations log in session state if not present
if 'transformations_log' not in st.session_state:
    st.session_state['transformations_log'] = []

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import eda_core
from src import auto_analyzer  
from src import report_builder 
from src.auto_analyzer import generate_cleaning_config, save_cleaning_config, load_cleaning_config, apply_cleaning_config

st.set_page_config(page_title="AutoEDA – AI-Powered Data Cleaner", layout="wide")
st.title("📊 AutoEDA – Smart Data Cleaning & Report Generator")


uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])


def display_data_summary(df):
    st.subheader("📌 Basic Information")
    if not df.empty:  
        
        basic_info = eda_core.get_basic_info(df)
        column_types = eda_core.classify_column_types(df)
        
        
        display_info = basic_info.copy()
        display_info["Classified Data Types"] = column_types
        
        st.json(display_info)
    else:
        st.warning("⚠️ DataFrame is empty.")

    st.subheader("📈 Summary Statistics")
    if not df.empty:  
        st.dataframe(eda_core.get_summary_statistics(df))
    else:
        st.warning("⚠️ DataFrame is empty.")

    st.subheader("🧬 Duplicate Rows")
    if not df.empty:  
        duplicates = eda_core.get_duplicate_info(df)
        st.write(f"🔁 Total duplicate rows: {duplicates}")
    else:
        st.warning("⚠️ DataFrame is empty.")


def display_missing_values(df):
    st.subheader("⚠️ Missing Values Summary")
    missing = eda_core.get_missing_values(df)
    if not missing.empty:
        st.dataframe(missing)

        st.markdown("#### Missing Value Visualizations")
        
        
        st.write("**Missing Values Matrix:**")
        
        # matrix_plot = msno.matrix(df, figsize=(10, 4))
        # matrix_plot_path = "reports/missingno_matrix.png"
        #
        # os.makedirs("reports", exist_ok=True)
        # matrix_plot.get_figure().savefig(matrix_plot_path)
        # st.image(matrix_plot_path, use_container_width=True)
        # plt.close(matrix_plot.get_figure()) 

       
        st.write("**Missing Values Bar Chart:**")
        # bar_plot = msno.bar(df, figsize=(10, 4))
        # bar_plot_path = "reports/missingno_bar.png"
        # bar_plot.get_figure().savefig(bar_plot_path)
        # st.image(bar_plot_path, use_container_width=True)
        # plt.close(bar_plot.get_figure()) 
        
        
        st.markdown("### 🧹 Handle Missing Data")
        
       
        suggestions = auto_analyzer.analyze_dataset(df)
        
        tab1, tab2, tab3 = st.tabs(["📊 AI Recommendations", "⚙️ Manual Settings", "📈 Preview"])
        
        with tab1:
            st.markdown("#### 🤖 AI-Powered Recommendations")
            for col, suggestion in suggestions.items():
                if col != "Outliers":  
                    st.markdown(f"**{col}**")
                    st.markdown(f"- Missing: {suggestion['missing_count']} values ({suggestion['missing_percent']:.1f}%)")
                    st.markdown(f"- Type: {suggestion['type']}")
                    st.markdown(f"- Recommendation: {suggestion['recommendation']}")
                    
                    if st.button(f"Apply Recommendation for {col}", key=f"apply_{col}"):
                        if "KNN" in suggestion['recommendation']:
                            strategy = 'knn'
                            df = auto_analyzer.handle_missing_values(df, strategy=strategy)
                        elif "mean" in suggestion['recommendation'].lower():
                            strategy = 'mean'
                            df = auto_analyzer.handle_missing_values(df, strategy=strategy)
                        elif "mode" in suggestion['recommendation'].lower():
                            strategy = 'mode'
                            df = auto_analyzer.handle_missing_values(df, strategy=strategy)
                        else:
                            strategy = 'auto'
                            df = auto_analyzer.handle_missing_values(df, strategy=strategy)
                        st.success(f"Applied recommendation for {col}")
                        st.session_state['transformations_log'].append({
                            'action_type': 'impute',
                            'columns': [col],
                            'parameters': {'strategy': strategy, 'custom_value': None},
                            'timestamp': datetime.datetime.now().isoformat()
                        })
        
        with tab2:
            st.markdown("#### ⚙️ Manual Settings")
            strategy = st.selectbox(
                "Select Imputation Strategy",
                ["auto", "knn", "mean", "median", "mode", "custom", "drop"],
                help="Choose how to handle missing values"
            )
            
            if strategy == "custom":
                custom_value = st.text_input("Enter custom value to fill missing cells:")
                if st.button("Apply Custom Value"):
                    df = auto_analyzer.handle_missing_values(df, strategy='custom', custom_value=custom_value)
                    st.success("Applied custom value imputation")
                    st.session_state['transformations_log'].append({
                        'action_type': 'impute',
                        'columns': [col],
                        'parameters': {'strategy': strategy, 'custom_value': custom_value},
                        'timestamp': datetime.datetime.now().isoformat()
                    })
            else:
                if st.button("Apply Strategy"):
                    df = auto_analyzer.handle_missing_values(df, strategy=strategy)
                    st.success(f"Applied {strategy} imputation")
                    st.session_state['transformations_log'].append({
                        'action_type': 'impute',
                        'columns': [col],
                        'parameters': {'strategy': strategy, 'custom_value': None},
                        'timestamp': datetime.datetime.now().isoformat()
                    })
        
        with tab3:
            st.markdown("#### 📈 Data Preview")
            st.dataframe(df.head())
            st.markdown("##### Missing Values After Imputation")
            missing_after = eda_core.get_missing_values(df)
            if missing_after.empty:
                st.success("✅ No missing values remaining!")
            else:
                st.dataframe(missing_after)
    else:
        st.success("✅ No missing values found!")
    
    return df


def display_outliers(df):
    st.subheader("🚨 Outlier Detection")
 
    suggestions = auto_analyzer.analyze_dataset(df)
    iso_forest_suggestion = suggestions.get("Outliers", None)

    if iso_forest_suggestion:
        st.info(f"**Isolation Forest:** {iso_forest_suggestion['message']}")
        st.info(f"**Isolation Forest Recommendation:** {iso_forest_suggestion['recommendation']}")
        
        st.markdown("#### Other Outlier Detection Methods")
        
        # IQR Detection
        outliers_iqr = eda_core.detect_outliers(df, output_dir="reports/boxplots") # This also saves boxplots
        if outliers_iqr:
            st.write("**IQR Method:**")
            for col, count in outliers_iqr.items():
                st.write(f"- Column '{col}': {count} outliers")
        else:
             st.info("**IQR Method:** No significant outliers detected.")

        # Z-Score Detection
        outliers_zscore = eda_core.detect_outliers_zscore(df, threshold=3) # Using default threshold 3
        if outliers_zscore:
            st.write("**Z-Score Method (Threshold=3):**")
            for col, count in outliers_zscore.items():
                 st.write(f"- Column '{col}': {count} outliers")
        else:
            st.info("**Z-Score Method:** No significant outliers detected.")

        st.markdown("### 📊 Boxplots")

        if not df.select_dtypes(include=np.number).empty:
             
            outliers_iqr_for_display = eda_core.detect_outliers(df, output_dir="reports/boxplots") # Rerun to ensure plots are saved if not already
            for col in outliers_iqr_for_display:
                 # img_path = f"reports/boxplots/boxplot_{col}.png"
                 # if os.path.exists(img_path):
                 #     st.image(img_path, caption=f"Boxplot for {col}", use_container_width=True)
                 pass
        else:
             st.warning("⚠️ No numerical columns found for outlier visualization.")

        st.markdown("### 🧹 Handle Outliers")
        
        tab1, tab2 = st.tabs(["⚙️ Manual Settings", "📈 Preview"])
        
        with tab1:
            st.markdown("#### ⚙️ Manual Settings")
            strategy = st.selectbox(
                "Select Outlier Handling Strategy",
                ['remove', 'replace_boundary', 'custom'],
                help="Choose how to handle detected outliers"
            )
            
            if strategy == "custom":
                custom_value = st.text_input("Enter custom value to replace outliers:")
                try:
                    custom_value = float(custom_value)
                except ValueError:
                    custom_value = None 
                    st.warning("Please enter a valid number for custom replacement.")

                if st.button("Apply Custom Value", key="apply_custom_outlier"):
                     if custom_value is not None:
                        df = auto_analyzer.handle_outliers(df, strategy='custom', custom_value=custom_value)
                        st.success("Applied custom value outlier handling")
                        st.session_state['transformations_log'].append({
                            'action_type': 'outlier_handling',
                            'columns': [col],
                            'parameters': {'strategy': strategy, 'custom_value': custom_value},
                            'timestamp': datetime.datetime.now().isoformat()
                        })
                     else:
                         st.error("Invalid custom value entered.")
                         
            else:
                if st.button("Apply Strategy", key="apply_selected_outlier_strategy"):
                    df = auto_analyzer.handle_outliers(df, strategy=strategy)
                    st.success(f"Applied {strategy} outlier handling")
        
        with tab2:
            st.markdown("#### 📈 Data Preview")
            st.dataframe(df.head())
            
            st.info("Preview shows the first few rows after outlier handling.")

    else:
        st.info("✅ No significant outliers detected by Isolation Forest.")
    
    return df


def display_visualizations(df):
    st.subheader("📊 Data Visualizations")
    
    
    output_dir = "reports/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    column_types = eda_core.classify_column_types(df)
    
    
    st.markdown("#### Histograms (Numerical Features)")
    num_cols = [col for col, col_type in column_types.items() if col_type == 'Numerical']
    if num_cols:
        for col in num_cols:
            eda_core.save_histogram(df, col, output_dir)
            # img_path = os.path.join(output_dir, f"histogram_{col}.png")
            # if os.path.exists(img_path):
            #     st.image(img_path, caption=f"Histogram of {col}", use_container_width=True)
    else:
        st.info("No numerical columns to display histograms.")

    
    st.markdown("#### Bar Charts (Categorical Features)")
    cat_cols = [col for col, col_type in column_types.items() if col_type == 'Categorical']
    if cat_cols:
        for col in cat_cols:
            eda_core.save_bar_chart(df, col, output_dir)
            # img_path = os.path.join(output_dir, f"bar_chart_{col}.png")
            # if os.path.exists(img_path):
            #      st.image(img_path, caption=f"Bar Chart of {col}", use_container_width=True)
    else:
        st.info("No categorical columns to display bar charts.")

    
    st.markdown("#### Correlation Heatmap")
    if num_cols and len(num_cols) > 1:
        eda_core.save_correlation_heatmap(df, output_dir)
        # img_path = os.path.join(output_dir, "correlation_heatmap.png")
        # if os.path.exists(img_path):
        #      st.image(img_path, caption="Correlation Heatmap", use_container_width=True)
    else:
        st.info("Not enough numerical columns (at least 2) to display a correlation heatmap.")

    # Pairplot
    st.markdown("#### Pairplot")
    if num_cols and len(num_cols) > 1:
        eda_core.save_pairplot(df, output_dir)
        # img_path = os.path.join(output_dir, "pairplot.png")
        # if os.path.exists(img_path):
        #      st.image(img_path, caption="Pairplot", use_container_width=True)
    else:
        st.info("Not enough numerical columns (at least 2) to display a pairplot.")


def display_custom_column_operations(df):
    st.subheader("⚙️ Custom Column-wise Operations")
    
    st.markdown("#### Drop Columns")
    
    all_columns = df.columns.tolist()
    columns_to_drop = st.multiselect(
        "Select columns to drop",
        all_columns,
        key='columns_to_drop_multiselect' # Unique key for the multiselect
    )
    
    if st.button("Drop Selected Columns", key='drop_columns_button'):
        if columns_to_drop:
            df = auto_analyzer.drop_columns(df, columns_to_drop)
            st.success(f"Successfully dropped columns: {columns_to_drop}")
        else:
            st.warning("Please select columns to drop.")
            
    st.markdown("#### Convert Column Data Type")
    
    all_columns = df.columns.tolist()
    
    col_to_convert = st.selectbox(
        "Select a column to convert",
        all_columns,
        key='col_to_convert_selectbox' # Unique key
    )
    
    # List of common pandas dtypes for conversion
    common_dtypes = ['int', 'float', 'str', 'datetime64']
    
    target_dtype = st.selectbox(
        "Select target data type",
        common_dtypes,
        key='target_dtype_selectbox' # Unique key
    )
    
    if st.button("Convert Data Type", key='convert_dtype_button'):
        if col_to_convert and target_dtype:
            df, message = auto_analyzer.convert_column_dtype(df, col_to_convert, target_dtype)
            if "Error" in message or "Warning" in message:
                st.warning(message) # Use warning for both errors and warnings from the function
            else:
                st.success(message)
        else:
            st.warning("Please select a column and target data type.")
            
    st.markdown("#### Rename Column")
    
    all_columns = df.columns.tolist()
    
    col_to_rename = st.selectbox(
        "Select a column to rename",
        all_columns,
        key='col_to_rename_selectbox' # Unique key
    )
    
    new_column_name = st.text_input(
        f"Enter new name for '{col_to_rename}'",
        key='new_col_name_textinput' # Unique key
    )
    
    if st.button("Rename Column", key='rename_column_button'):
        if col_to_rename and new_column_name:
            df, message = auto_analyzer.rename_column(df, col_to_rename, new_column_name)
            if "Error" in message:
                st.error(message)
            else:
                st.success(message)
        else:
            st.warning("Please select a column and enter a new name.")
            
    return df


def display_cleaning_configuration(df):
    """Display cleaning configuration options."""
    st.subheader("🔄 Cleaning Configuration")
    
    # Create tabs for different configuration options
    config_tab1, config_tab2 = st.tabs(["💾 Save Configuration", "📂 Load Configuration"])
    
    with config_tab1:
        st.write("Save your current cleaning configuration")
        config_name = st.text_input("Configuration Name", key="config_name")
        if st.button("Save Configuration", key="save_config"):
            if config_name:
                config = generate_cleaning_config(df, st.session_state)
                success, message = save_cleaning_config(config, f"configs/{config_name}.json")
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Please enter a configuration name")
    
    with config_tab2:
        st.write("Load a saved cleaning configuration")
        # List available configurations
        import os
        config_dir = "configs"
        os.makedirs(config_dir, exist_ok=True)
        config_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]
        
        if config_files:
            selected_config = st.selectbox("Select Configuration", config_files, key="config_select")
            if st.button("Load Configuration", key="load_config"):
                config, message = load_cleaning_config(os.path.join(config_dir, selected_config))
                if config:
                    st.session_state['loaded_config'] = config
                    st.success(message)
                    # Apply the loaded configuration
                    df_cleaned, apply_message = apply_cleaning_config(df, config)
                    st.session_state['df'] = df_cleaned
                    st.success(apply_message)
                else:
                    st.error(message)
        else:
            st.info("No saved configurations found")


def export_cleaned_data(df):
    st.subheader("📥 Download Cleaned Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="cleaned_data.csv",
                      mime="text/csv")


def export_html_report_section(df):
    st.subheader("📄 Export/Download Full EDA & Cleaning Report")
    if st.button("Generate HTML Report", key="generate_html_report"):
        report_data = report_builder.generate_eda_report_data(df)
        transformations_log = st.session_state.get('transformations_log', [])
        html_path = report_builder.build_html_report(report_data, transformations_log)
        st.session_state['last_html_report_path'] = html_path
        st.success(f"Report generated: {html_path}")
    if 'last_html_report_path' in st.session_state:
        with open(st.session_state['last_html_report_path'], 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.download_button(
            label="Download HTML Report",
            data=html_content,
            file_name="AutoEDA_Report.html",
            mime="text/html"
        )
        # PDF Export Section
        if st.button("Generate PDF Report", key="generate_pdf_report"):
            pdf_path = report_builder.build_pdf_report(st.session_state['last_html_report_path'])
            st.session_state['last_pdf_report_path'] = pdf_path
            st.success(f"PDF report generated: {pdf_path}")
        if 'last_pdf_report_path' in st.session_state:
            with open(st.session_state['last_pdf_report_path'], 'rb') as f:
                pdf_content = f.read()
            st.download_button(
                label="Download PDF Report",
                data=pdf_content,
                file_name="AutoEDA_Report.pdf",
                mime="application/pdf"
            )


if uploaded_file:
    try:  # Add try-except for error handling
        filepath = os.path.join("data", uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        df = pd.read_csv(filepath)
        st.success("✅ File uploaded successfully!")

        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head())
        st.write(f"🧮 Shape: {df.shape}")

        # Train AI Models immediately after file upload
        with st.spinner("Training AI Models..."):
            auto_analyzer.train_models(df.copy())  # Train models on the uploaded data
        st.success("✅ AI Models Trained!")

        # Generate all visualization images for report (but do not display in UI)
        output_dir = "reports/visualizations"
        os.makedirs(output_dir, exist_ok=True)
        boxplot_dir = "reports/boxplots"
        os.makedirs(boxplot_dir, exist_ok=True)
        # Generate histograms and boxplots for numerical columns
        num_cols = [col for col, col_type in eda_core.classify_column_types(df).items() if col_type == 'Numerical']
        for col in num_cols:
            eda_core.save_histogram(df, col, output_dir)
            eda_core.save_boxplot(df, col, boxplot_dir)
        # Generate bar charts for categorical columns
        cat_cols = [col for col, col_type in eda_core.classify_column_types(df).items() if col_type == 'Categorical']
        for col in cat_cols:
            eda_core.save_bar_chart(df, col, output_dir)
        # Generate heatmap and pairplot if enough numerical columns
        if len(num_cols) > 1:
            eda_core.save_correlation_heatmap(df, output_dir)
            eda_core.save_pairplot(df, output_dir)
        # Generate missingno plots
        matrix_plot = msno.matrix(df, figsize=(10, 4))
        matrix_plot_path = "reports/missingno_matrix.png"
        matrix_plot.get_figure().savefig(matrix_plot_path)
        plt.close(matrix_plot.get_figure())
        bar_plot = msno.bar(df, figsize=(10, 4))
        bar_plot_path = "reports/missingno_bar.png"
        bar_plot.get_figure().savefig(bar_plot_path)
        plt.close(bar_plot.get_figure())

        # Generate EDA Report Data
        st.subheader("📊 EDA Report")
        eda_report_data = report_builder.generate_eda_report_data(df)

        # Display EDA Report sections from the collected data
        st.markdown("### Basic Information")
        st.json(eda_report_data.get('basic_info', {}))
        st.markdown("### Classified Column Types")
        st.json(eda_report_data.get('column_types', {}))
        st.markdown("### Summary Statistics and Data Quality")
        st.dataframe(pd.DataFrame.from_dict(eda_report_data.get('summary_statistics', {}), orient='index'))
        st.markdown("### Value Density Analysis")
        density_summaries = eda_report_data.get('density_summaries', {})
        if density_summaries:
            for col, summary in density_summaries.items():
                st.write(f"**{col}:** {summary}")
        else:
            st.info("No numerical columns found for density analysis or no clear density ranges detected.")

        st.markdown("### Outlier Detection Summary")
        st.write("**Isolation Forest:**")
        iso_forest_outliers = eda_report_data.get('outliers_isolation_forest', None)
        if iso_forest_outliers:
             st.info(iso_forest_outliers.get('message', 'N/A'))
        else:
             st.info("Isolation Forest: No significant outliers detected.")
        
        st.write("**IQR Method:**")
        iqr_outliers = eda_report_data.get('outliers_iqr', {})
        if iqr_outliers:
            for col, count in iqr_outliers.items():
                st.write(f"- Column '{col}': {count} outliers")
        else:
             st.info("IQR Method: No significant outliers detected.")
        
        st.write("**Z-Score Method (Threshold=3):**")
        zscore_outliers = eda_report_data.get('outliers_zscore', {})
        if zscore_outliers:
            for col, count in zscore_outliers.items():
                 st.write(f"- Column '{col}': {count} outliers")
        else:
            st.info("Z-Score Method: No significant outliers detected.")

        # Display Visualizations (still generated and displayed separately as they are images)
        display_missing_values(df) # Restore missing value handling controls in the UI
        df = display_outliers(df) # This now only displays boxplots and handle outliers section, and returns the potentially modified df
        # display_visualizations(df)

        # Display Custom Column Operations
        df = display_custom_column_operations(df) # Call the new custom operations function

        # Display Cleaning Configuration
        display_cleaning_configuration(df)

        # Handle Cleaning (These sections remain interactive for user input)
        # The interactive handling sections are kept separate from the main report display

        # AI-Powered Suggestions (These are now part of the interactive handling sections)
        # The separate display of AI suggestions is removed as they are shown within the handling tabs

        export_cleaned_data(df)
        export_html_report_section(df)

        # Dedicated Visualizations Tab/Expander
        with st.expander("📊 Show All Visualizations (New Tab)"):
            st.markdown("### All Visualizations")
            viz_tab = st.tabs(["Visualizations"])[0]
            with viz_tab:
                # Boxplots
                st.subheader("Boxplots")
                for img in sorted(glob.glob("reports/boxplots/boxplot_*.png")):
                    st.image(img, caption=img.split("/")[-1], use_column_width=True)

                # Histograms
                st.subheader("Histograms")
                for img in sorted(glob.glob("reports/visualizations/histogram_*.png")):
                    st.image(img, caption=img.split("/")[-1], use_column_width=True)

                # Bar Charts
                st.subheader("Bar Charts")
                for img in sorted(glob.glob("reports/visualizations/bar_chart_*.png")):
                    st.image(img, caption=img.split("/")[-1], use_column_width=True)

                # Missingno Plots
                st.subheader("Missing Value Visualizations")
                for img in ["reports/missingno_matrix.png", "reports/missingno_bar.png"]:
                    if os.path.exists(img):
                        st.image(img, caption=img.split("/")[-1], use_column_width=True)

                # Heatmap
                st.subheader("Correlation Heatmap")
                heatmap_img = "reports/visualizations/correlation_heatmap.png"
                if os.path.exists(heatmap_img):
                    st.image(heatmap_img, caption="correlation_heatmap.png", use_column_width=True)

                # Pairplot
                st.subheader("Pairplot")
                pairplot_img = "reports/visualizations/pairplot.png"
                if os.path.exists(pairplot_img):
                    st.image(pairplot_img, caption="pairplot.png", use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("⬆️ Upload a CSV file to begin.")