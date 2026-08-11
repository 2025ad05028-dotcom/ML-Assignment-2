import streamlit as st
import pandas as pd

st.title("Breast Cancer Classification Models")

st.write("Machine Learning Assignment 2")

metrics = pd.read_csv("metrics.csv")

st.subheader("Model Performance Comparison")
st.dataframe(metrics)

selected_model = st.selectbox(
    "Select a Model",
    metrics["Model"]
)

selected_row = metrics[
    metrics["Model"] == selected_model
]

st.subheader("Selected Model Metrics")
st.write(selected_row)

uploaded_file = st.file_uploader(
    "Upload Test Data CSV",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")

    st.dataframe(data.head())