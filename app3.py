import streamlit as st
import pandas as pd
import pickle
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

def load_resource(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Function to get user input from sidebar
def get_user_input():
    inputs = {
        'TotalCharges': st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=0.0),
        'MonthlyCharges': st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=0.0),
        'tenure': st.sidebar.slider("Tenure (months)", 0, 100, 0),
        'Contract': st.sidebar.selectbox("Contract", options=[0, 1, 2], format_func=lambda x: contract_options[x]),
        'OnlineSecurity': st.sidebar.selectbox("Online Security", options=[0, 1, 2], format_func=lambda x: online_security_options[x]),
        'PaymentMethod': st.sidebar.selectbox("Payment Method", options=[0, 1, 2, 3], format_func=lambda x: payment_method_options[x]),
        'TechSupport': st.sidebar.selectbox("Tech Support", options=[0, 1, 2], format_func=lambda x: tech_support_options[x]),
        'InternetService': st.sidebar.selectbox("Internet Service", options=[0, 1, 2], format_func=lambda x: internet_service_options[x]),
        'PaperlessBilling': st.sidebar.selectbox("Paperless Billing", options=[0, 1], format_func=lambda x: paperless_billing_options[x]),
        'DeviceProtection': st.sidebar.selectbox("Device Protection", options=[0, 1, 2], format_func=lambda x: device_protection_options[x])
    }
    return pd.DataFrame([inputs])


def predict_churn(model, scaler, inputs):
    scaled_inputs = scaler.transform(inputs)
    return model.predict(scaled_inputs)

# UI Titles
st.title("Customer Churn Prediction Tool")
st.sidebar.header("Input Parameters")

# Load scaler and model
scaler = load_resource('Scaled.pkl')
model = load_model('optimal_model.h5')

# Mappings for categorical features
contract_options = {0: 'Month-to-month', 1: 'One year', 2: 'Two years'}
online_security_options = {0: 'No', 1: 'No internet service', 2: 'Yes'}
payment_method_options = {0: 'Electronic check', 1: 'Mailed check', 2: 'Bank transfer', 3: 'Credit card'}
tech_support_options = {0: 'No', 1: 'No internet service', 2: 'Yes'}
internet_service_options = {0: 'DSL', 1: 'Fiber optic', 2: 'No'}
paperless_billing_options = {0: 'No', 1: 'Yes'}
device_protection_options = {0: 'No', 1: 'No internet service', 2: 'Yes'}

# User input
input_data = get_user_input()


if st.sidebar.button("Predict"):
    churn_prediction = predict_churn(model, scaler, input_data)
    prediction_label = "Churn" if churn_prediction[0, 0] > 0.5 else "No Churn"
    confidence = churn_prediction[0, 0]

    st.subheader("Prediction Results:")
    st.write(f"Prediction: **{prediction_label}**")
    st.write(f"Confidence Score: **{confidence:.2f}**")
