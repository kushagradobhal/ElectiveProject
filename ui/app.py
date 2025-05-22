import streamlit as st
import pandas as pd
import os
import sys
import numpy as np

# Consider using relative imports or packaging for better structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import eda_core
from src import auto_analyzer  # Import auto_analyzer

st.set_page_config(page_title="AutoEDA – AI-Powered Data Cleaner", layout="wide")
st.title("📊 AutoEDA – Smart Data Cleaning & Report Generator")

# File Upload
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])


def display_data_summary(df):
    st.subheader("📌 Basic Information")
    if not df.empty:  # Check if DataFrame is empty
        st.json(eda_core.get_basic_info(df))
    else:
        st.warning("⚠️ DataFrame is empty.")

    st.subheader("📈 Summary Statistics")
    if not df.empty:  # Check if DataFrame is empty
        st.dataframe(eda_core.get_summary_statistics(df))
    else:
        st.warning("⚠️ DataFrame is empty.")

    st.subheader("🧬 Duplicate Rows")
    if not df.empty:  # Check if DataFrame is empty
        duplicates = eda_core.get_duplicate_info(df)
        st.write(f"🔁 Total duplicate rows: {duplicates}")
    else:
        st.warning("⚠️ DataFrame is empty.")


def display_missing_values(df):
    st.subheader("⚠️ Missing Values Summary")
    missing = eda_core.get_missing_values(df)
    if not missing.empty:
        st.dataframe(missing)
        
        st.markdown("### 🧹 Handle Missing Data")
        
        # Get AI suggestions
        suggestions = auto_analyzer.analyze_dataset(df)
        
        # Create tabs for different handling strategies
        tab1, tab2, tab3 = st.tabs(["📊 AI Recommendations", "⚙️ Manual Settings", "📈 Preview"])
        
        with tab1:
            st.markdown("#### 🤖 AI-Powered Recommendations")
            for col, suggestion in suggestions.items():
                if col != "Outliers":  # Skip outlier suggestions
                    st.markdown(f"**{col}**")
                    st.markdown(f"- Missing: {suggestion['missing_count']} values ({suggestion['missing_percent']:.1f}%)")
                    st.markdown(f"- Type: {suggestion['type']}")
                    st.markdown(f"- Recommendation: {suggestion['recommendation']}")
                    
                    # Add apply button for each recommendation
                    if st.button(f"Apply Recommendation for {col}", key=f"apply_{col}"):
                        if "KNN" in suggestion['recommendation']:
                            df = auto_analyzer.handle_missing_values(df, strategy='knn')
                        elif "mean" in suggestion['recommendation'].lower():
                            df = auto_analyzer.handle_missing_values(df, strategy='mean')
                        elif "mode" in suggestion['recommendation'].lower():
                            df = auto_analyzer.handle_missing_values(df, strategy='mode')
                        st.success(f"Applied recommendation for {col}")
        
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
            else:
                if st.button("Apply Strategy"):
                    df = auto_analyzer.handle_missing_values(df, strategy=strategy)
                    st.success(f"Applied {strategy} imputation")
        
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
    
    # Get AI suggestions for outliers
    suggestions = auto_analyzer.analyze_dataset(df)
    outlier_suggestion = suggestions.get("Outliers", None)

    if outlier_suggestion:
        st.info(outlier_suggestion['message'])
        st.info(outlier_suggestion['recommendation'])

        st.markdown("### 📊 Boxplots")
        # Outlier detection and boxplot saving (using IQR as before)
        # This part still uses the IQR method for visualization purposes
        if not df.select_dtypes(include=np.number).empty:
            outliers_iqr = eda_core.detect_outliers(df, output_dir="reports/boxplots") # This is for visualization
            for col in outliers_iqr:
                img_path = f"reports/boxplots/boxplot_{col}.png"
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"Boxplot for {col}", use_column_width=True)
        else:
            st.warning("⚠️ No numerical columns found for outlier visualization.")

        st.markdown("### 🧹 Handle Outliers")
        
        # Create tabs for different handling strategies
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
                # Ensure custom_value is a valid number if replacing numerical outliers
                try:
                    custom_value = float(custom_value)
                except ValueError:
                    custom_value = None # Or handle as an error
                    st.warning("Please enter a valid number for custom replacement.")

                if st.button("Apply Custom Value", key="apply_custom_outlier"):
                     if custom_value is not None:
                        df = auto_analyzer.handle_outliers(df, strategy='custom', custom_value=custom_value)
                        st.success("Applied custom value outlier handling")
                     else:
                         st.error("Invalid custom value entered.")
                         
            else:
                if st.button("Apply Strategy", key="apply_selected_outlier_strategy"):
                    df = auto_analyzer.handle_outliers(df, strategy=strategy)
                    st.success(f"Applied {strategy} outlier handling")
        
        with tab2:
            st.markdown("#### 📈 Data Preview")
            st.dataframe(df.head())
            # Optionally show outlier detection results again after handling
            # This might require re-running detection on the modified DF
            # For now, just show the head
            st.info("Preview shows the first few rows after outlier handling.")

    else:
        st.info("✅ No significant outliers detected by Isolation Forest.")
    
    return df


def export_cleaned_data(df):
    st.subheader("📥 Download Cleaned Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="cleaned_data.csv",
                      mime="text/csv")


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

        display_data_summary(df)
        
        # Train AI Models immediately after file upload
        with st.spinner("Training AI Models..."):
            auto_analyzer.train_models(df.copy())  # Train models on the uploaded data
        st.success("✅ AI Models Trained!")

        df = display_missing_values(df)
        df = display_outliers(df)

        # AI-Powered Suggestions
        st.subheader("🤖 AI-Powered Cleaning Suggestions")
        # Removed the checkbox as training is now automatic
        # train_models = st.checkbox("Train AI Models")  

        # suggestions = auto_analyzer.analyze_dataset(df)  # Get suggestions - This is now called inside display_missing_values

        # Removed the code block that iterated through suggestions here
        # It is now handled within the display_missing_values function
        # if suggestions:
        #     for col, suggestion in suggestions.items():
        #         st.write(f"**Column: {col}**")
        #         if isinstance(suggestion, dict):  # Check if it's a column suggestion
        #             st.write(f"- Issue: {suggestion['missing']}")
        #             st.write(f"- Recommendation: {suggestion['recommendation']}")
        #         else:  # It's the general "Outliers" suggestion
        #             st.write(f"- Suggestion: {suggestion}")
        # else:
        #     st.info("No AI-powered suggestions at this time.")

        export_cleaned_data(df)

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("⬆️ Upload a CSV file to begin.")