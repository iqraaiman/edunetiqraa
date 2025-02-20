
import pickle
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score

#Set page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕")

# Load the saved diabetes model
diabetes_model_path = r"C:\Users\Sharaf\Desktop\project diabetes\diabetes_model.sav"
diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))

# Page title
st.title('Diabetes Prediction using ML')

# Gettinf the input data from the user
col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.text_input('Number of Pregnancies')

with col2:
    Glucose = st.text_input('Glucose Level')

with col1:
    BloodPressure = st.text_input('Blood Pressure')

with col2:
    SkinThickness = st.text_input('Skin Thickness')

with col1:
    Insulin = st.text_input('Insulin Level')

with col2:
    BMI = st.text_input('Body Mass Index')

with col1:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')

with col2:
    Age = st.text_input('Age')

diab_diagnosis = ''

if st.button('Diabetes Test Result'):
    try:
        # Convert input to float
        user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), 
                      float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
        
        # Make prediction
        diab_prediction = diabetes_model.predict([user_input])

        # Display result
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

        st.success(diab_diagnosis)
    
    except ValueError:
        st.error("Please enter valid numerical values for all fields.")