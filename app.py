import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os
from PIL import Image
import time

# Configuration
MODEL_URL = "https://drive.google.com/uc?id=134MQZya_pUW1DB8H6i_ehzYilZ8fjzqm"
MODEL_PATH = "credit_card_default_model.pkl"

@st.cache_resource
def download_model():
    """Download the model file from Google Drive if not already present"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading model... (this may take a few minutes)'):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

# Load model (will automatically download if needed)
try:
    model = download_model()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

def predict_default(input_data):
    """Predict default for a single record"""
    input_df = pd.DataFrame([input_data])
    
    # Calculate derived features
    pay_columns = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    pay_amt_columns = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    bill_amt_columns = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    
    input_df['AVG_PAY_DELAY'] = input_df[pay_columns].mean(axis=1)
    input_df['TOTAL_PAY_AMT'] = input_df[pay_amt_columns].sum(axis=1)
    input_df['TOTAL_BILL_AMT'] = input_df[bill_amt_columns].sum(axis=1)
    input_df['CREDIT_UTILIZATION'] = input_df['TOTAL_BILL_AMT'] / input_df['LIMIT_BAL']
    input_df['PAYMENT_TO_BILL_RATIO'] = input_df['TOTAL_PAY_AMT'] / (input_df['TOTAL_BILL_AMT'] + 1)
    input_df['SEVERE_DELINQUENCY'] = (input_df[pay_columns] >= 3).any(axis=1).astype(int)
    
    # Select final features
    features = ['LIMIT_BAL', 'AGE', 'AVG_PAY_DELAY', 'TOTAL_PAY_AMT', 
                'TOTAL_BILL_AMT', 'CREDIT_UTILIZATION', 'PAYMENT_TO_BILL_RATIO',
                'SEVERE_DELINQUENCY']
    
    # Make prediction only when explicitly called
    prediction = model.predict(input_df[features])
    probability = model.predict_proba(input_df[features])[0][1]
    
    return prediction[0], probability

def main():
    st.title("Credit Card Default Risk Predictor")
    st.write("Predict the likelihood of credit card default based on payment history and bill amounts.")
    
    # Initialize session state for prediction results
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    # Input form
    with st.form("prediction_form"):
        st.header("Customer Information")
        col1, col2 = st.columns(2)
        with col1:
            limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0, value=20000)
        with col2:
            age = st.number_input("Age (AGE)", min_value=18, max_value=100, value=30)
        
        st.header("Payment History (Delays in Months)")
        cols = st.columns(6)
        with cols[0]:
            pay_0 = st.number_input("September (PAY_0)", min_value=-2, max_value=8, value=0)
        with cols[1]:
            pay_2 = st.number_input("August (PAY_2)", min_value=-2, max_value=8, value=0)
        with cols[2]:
            pay_3 = st.number_input("July (PAY_3)", min_value=-2, max_value=8, value=0)
        with cols[3]:
            pay_4 = st.number_input("June (PAY_4)", min_value=-2, max_value=8, value=0)
        with cols[4]:
            pay_5 = st.number_input("May (PAY_5)", min_value=-2, max_value=8, value=0)
        with cols[5]:
            pay_6 = st.number_input("April (PAY_6)", min_value=-2, max_value=8, value=0)
        
        st.header("Payment Amounts")
        cols = st.columns(6)
        with cols[0]:
            pay_amt1 = st.number_input("September (PAY_AMT1)", min_value=0, value=1000)
        with cols[1]:
            pay_amt2 = st.number_input("August (PAY_AMT2)", min_value=0, value=1000)
        with cols[2]:
            pay_amt3 = st.number_input("July (PAY_AMT3)", min_value=0, value=1000)
        with cols[3]:
            pay_amt4 = st.number_input("June (PAY_AMT4)", min_value=0, value=1000)
        with cols[4]:
            pay_amt5 = st.number_input("May (PAY_AMT5)", min_value=0, value=1000)
        with cols[5]:
            pay_amt6 = st.number_input("April (PAY_AMT6)", min_value=0, value=1000)
        
        st.header("Bill Statements")
        cols = st.columns(6)
        with cols[0]:
            bill_amt1 = st.number_input("September (BILL_AMT1)", min_value=0, value=2000)
        with cols[1]:
            bill_amt2 = st.number_input("August (BILL_AMT2)", min_value=0, value=2000)
        with cols[2]:
            bill_amt3 = st.number_input("July (BILL_AMT3)", min_value=0, value=2000)
        with cols[3]:
            bill_amt4 = st.number_input("June (BILL_AMT4)", min_value=0, value=2000)
        with cols[4]:
            bill_amt5 = st.number_input("May (BILL_AMT5)", min_value=0, value=2000)
        with cols[5]:
            bill_amt6 = st.number_input("April (BILL_AMT6)", min_value=0, value=2000)
        
        submitted = st.form_submit_button("Predict Default Risk")
    
    # Only make prediction after button is clicked
    if submitted:
        with st.spinner('Making prediction...'):
            time.sleep(1)  # Simulate processing time
            
            input_data = {
                'LIMIT_BAL': limit_bal,
                'AGE': age,
                'PAY_0': pay_0,
                'PAY_2': pay_2,
                'PAY_3': pay_3,
                'PAY_4': pay_4,
                'PAY_5': pay_5,
                'PAY_6': pay_6,
                'PAY_AMT1': pay_amt1,
                'PAY_AMT2': pay_amt2,
                'PAY_AMT3': pay_amt3,
                'PAY_AMT4': pay_amt4,
                'PAY_AMT5': pay_amt5,
                'PAY_AMT6': pay_amt6,
                'BILL_AMT1': bill_amt1,
                'BILL_AMT2': bill_amt2,
                'BILL_AMT3': bill_amt3,
                'BILL_AMT4': bill_amt4,
                'BILL_AMT5': bill_amt5,
                'BILL_AMT6': bill_amt6
            }
            
            try:
                prediction, probability = predict_default(input_data)
                st.session_state.prediction_made = True
                st.session_state.prediction_result = (prediction, probability)
                st.session_state.input_data = input_data
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    # Show results only after prediction
    if st.session_state.prediction_made:
        st.subheader("Prediction Results")
        prediction, probability = st.session_state.prediction_result
        
        if prediction == 1:
            st.error(f"🚨 High Risk of Default (Probability: {probability:.2%})")
            
            # Explain risk factors
            st.write("### Key Risk Factors:")
            avg_delay = np.mean([
                st.session_state.input_data['PAY_0'],
                st.session_state.input_data['PAY_2'],
                st.session_state.input_data['PAY_3'],
                st.session_state.input_data['PAY_4'],
                st.session_state.input_data['PAY_5'],
                st.session_state.input_data['PAY_6']
            ])
            
            if avg_delay > 0:
                st.write(f"- ⏱️ Average payment delay: {avg_delay:.1f} months")
            
            total_pay = sum([
                st.session_state.input_data['PAY_AMT1'],
                st.session_state.input_data['PAY_AMT2'],
                st.session_state.input_data['PAY_AMT3'],
                st.session_state.input_data['PAY_AMT4'],
                st.session_state.input_data['PAY_AMT5'],
                st.session_state.input_data['PAY_AMT6']
            ])
            
            total_bill = sum([
                st.session_state.input_data['BILL_AMT1'],
                st.session_state.input_data['BILL_AMT2'],
                st.session_state.input_data['BILL_AMT3'],
                st.session_state.input_data['BILL_AMT4'],
                st.session_state.input_data['BILL_AMT5'],
                st.session_state.input_data['BILL_AMT6']
            ])
            
            if total_pay < total_bill:
                st.write(f"- 💰 Payments ({total_pay}) less than bills ({total_bill})")
            
            credit_util = total_bill / st.session_state.input_data['LIMIT_BAL']
            if credit_util > 0.7:
                st.write(f"- 🏦 High credit utilization ({credit_util:.0%} of limit)")
            
        else:
            st.success(f"✅ Low Risk of Default (Probability: {probability:.2%})")

if __name__ == "__main__":
    main()
