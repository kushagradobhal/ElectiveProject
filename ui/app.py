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
    else:
        st.success("✅ No missing values found!")

    st.markdown("### 🧹 Handle Missing Data")
    option = st.radio("Choose cleaning method", ["None", "Drop rows", "Fill with value"])
    if option == "Drop rows":
        df = eda_core.clean_missing_values(df, method='drop')
        st.success("✅ Dropped rows with missing values.")
    elif option == "Fill with value":
        fill_value = st.text_input("Enter value to fill missing cells:")
        if fill_value:
            df = eda_core.clean_missing_values(df, method='fill', fill_value=fill_value)
            st.success("✅ Missing values filled.")
    return df


def display_outliers(df):
    st.subheader("🚨 Outlier Detection")
    if not df.select_dtypes(include=np.number).empty:  # Check for numerical columns
        outliers = eda_core.detect_outliers(df, output_dir="reports/boxplots")
        if outliers:
            st.json(outliers)
        else:
            st.info("✅ No outliers found.")

        st.markdown("### 📊 Boxplots")
        for col in outliers:
            img_path = f"reports/boxplots/boxplot_{col}.png"
            if os.path.exists(img_path):
                st.image(img_path, caption=f"Boxplot for {col}", use_column_width=True)
    else:
        st.warning("⚠️ No numerical columns found for outlier detection.")


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
        df = display_missing_values(df)
        display_outliers(df)

        # AI-Powered Suggestions
        st.subheader("🤖 AI-Powered Cleaning Suggestions")
        train_models = st.checkbox("Train AI Models")  # Add checkbox

        if train_models:
            with st.spinner("Training AI Models..."):  # Show spinner
                auto_analyzer.train_models(df.copy())  # Train models
            st.success("✅ AI Models Trained!")

        suggestions = auto_analyzer.analyze_dataset(df)  # Get suggestions

        if suggestions:
            for col, suggestion in suggestions.items():
                st.write(f"**Column: {col}**")
                if isinstance(suggestion, dict):  # Check if it's a column suggestion
                    st.write(f"- Issue: {suggestion['missing']}")
                    st.write(f"- Recommendation: {suggestion['recommendation']}")
                else:  # It's the general "Outliers" suggestion
                    st.write(f"- Suggestion: {suggestion}")
        else:
            st.info("No AI-powered suggestions at this time.")

        export_cleaned_data(df)

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("⬆️ Upload a CSV file to begin.")