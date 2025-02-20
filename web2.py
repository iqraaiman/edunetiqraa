import pickle
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score

# Set dark mode page title and layout
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🩺")

# Load the trained model
diabetes_model_path = r"C:\Users\Sharaf\Desktop\project diabetes\diabetes_model1.sav"

try:
    with open(diabetes_model_path, "rb") as model_file:
        diabetes_model = pickle.load(model_file)
except Exception as e:
    st.error(f"⚠ Error loading the model: {e}")

# Load dataset for real-time accuracy calculation
dataset_path = r"C:\Users\Sharaf\Desktop\project diabetes\diabetes.csv"

try:
    data = pd.read_csv(dataset_path)
    X = data.drop(columns=["Outcome"])  # Features
    y = data["Outcome"]  # Target (0: Non-Diabetic, 1: Diabetic)

    # Calculate model accuracy
    y_pred = diabetes_model.predict(X)
    real_time_accuracy = accuracy_score(y, y_pred)

    # Display accuracy in Streamlit
    st.write(f"### Model Accuracy: {real_time_accuracy*100:.2f}%")
except Exception as e:
    st.error(f"⚠ Error loading dataset: {e}")
