import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the data
@st.cache_data 
def load_data(file_path):
    return pd.read_csv(file_path)
def Exploration_Data():
    st.title("Data Exploration App")

    # Upload file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)

        st.subheader("Data Overview")
        st.write(df.head())

        st.subheader("Data Statistics")
        st.write(df.describe())

        # Choose column for histogram
        column_for_histogram = st.selectbox("Select a column for histogram", df.columns)
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column_for_histogram], kde=True)
        st.subheader(f"Histogram for {column_for_histogram}")
        st.pyplot()

        # Regression Plot
        column_for_histogram = st.selectbox("Select a column for Regression", df.columns)
        plt.figure(figsize=(10, 6))
        sns.regplot(x=df[column_for_histogram], y='Petrol_Consumption', data=df)
        st.subheader(f"Regression for {column_for_histogram}")
        st.pyplot()

        # Choose columns for correlation plot
        st.subheader("Correlation Plot")
        columns_for_correlation = st.multiselect("Select columns for correlation plot", df.columns)
        if columns_for_correlation:
            correlation_matrix = df[columns_for_correlation].corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="viridis", linewidths=.5)
            st.pyplot()

        # Scatter plot
        st.subheader("Scatter Plot")
        x_variable = st.selectbox("Select independent variable (X)", df.columns)
        y_variable = st.selectbox("Select dependent variable (Y)", df.columns)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[x_variable], y=df[y_variable])
        st.subheader(f"Scatter Plot between {x_variable} and {y_variable}")
        st.pyplot()

