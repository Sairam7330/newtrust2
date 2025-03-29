import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('credit_card_default_model.pkl')

def predict_default(input_data):
    # Create features from input data
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
    
    # Make prediction
    prediction = model.predict(input_df[features])
    probability = model.predict_proba(input_df[features])[0][1]
    
    return prediction[0], probability

def main():
    st.title("Credit Card Default Risk Predictor")
    st.write("This app predicts the likelihood of a credit card holder defaulting on their next payment.")
    
    # Input form
    with st.form("input_form"):
        st.header("Customer Information")
        limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0)
        age = st.number_input("Age (AGE)", min_value=18, max_value=100)
        
        st.header("Payment History (Delays in Months)")
        pay_0 = st.number_input("September Payment Delay (PAY_0)", min_value=-2, max_value=8)
        pay_2 = st.number_input("August Payment Delay (PAY_2)", min_value=-2, max_value=8)
        pay_3 = st.number_input("July Payment Delay (PAY_3)", min_value=-2, max_value=8)
        pay_4 = st.number_input("June Payment Delay (PAY_4)", min_value=-2, max_value=8)
        pay_5 = st.number_input("May Payment Delay (PAY_5)", min_value=-2, max_value=8)
        pay_6 = st.number_input("April Payment Delay (PAY_6)", min_value=-2, max_value=8)
        
        st.header("Payment Amounts")
        pay_amt1 = st.number_input("September Payment Amount (PAY_AMT1)", min_value=0)
        pay_amt2 = st.number_input("August Payment Amount (PAY_AMT2)", min_value=0)
        pay_amt3 = st.number_input("July Payment Amount (PAY_AMT3)", min_value=0)
        pay_amt4 = st.number_input("June Payment Amount (PAY_AMT4)", min_value=0)
        pay_amt5 = st.number_input("May Payment Amount (PAY_AMT5)", min_value=0)
        pay_amt6 = st.number_input("April Payment Amount (PAY_AMT6)", min_value=0)
        
        st.header("Bill Statements")
        bill_amt1 = st.number_input("September Bill Amount (BILL_AMT1)", min_value=0)
        bill_amt2 = st.number_input("August Bill Amount (BILL_AMT2)", min_value=0)
        bill_amt3 = st.number_input("July Bill Amount (BILL_AMT3)", min_value=0)
        bill_amt4 = st.number_input("June Bill Amount (BILL_AMT4)", min_value=0)
        bill_amt5 = st.number_input("May Bill Amount (BILL_AMT5)", min_value=0)
        bill_amt6 = st.number_input("April Bill Amount (BILL_AMT6)", min_value=0)
        
        submitted = st.form_submit_button("Predict Default Risk")
    
    if submitted:
        # Prepare input data
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
        
        # Make prediction
        prediction, probability = predict_default(input_data)
        
        # Display results
        st.subheader("Prediction Results")
        if prediction == 1:
            st.error(f"High Risk of Default (Probability: {probability:.2%})")
            
            # Explain key risk factors
            st.write("Key Risk Factors Identified:")
            
            # Payment delays
            if input_data['AVG_PAY_DELAY'] > 0:
                st.write(f"- Average payment delay of {input_data['AVG_PAY_DELAY']:.1f} months")
            
            # Payment to bill ratio
            if input_data['PAY_AMT1'] + input_data['PAY_AMT2'] + input_data['PAY_AMT3'] + \
               input_data['PAY_AMT4'] + input_data['PAY_AMT5'] + input_data['PAY_AMT6'] < \
               input_data['BILL_AMT1'] + input_data['BILL_AMT2'] + input_data['BILL_AMT3'] + \
               input_data['BILL_AMT4'] + input_data['BILL_AMT5'] + input_data['BILL_AMT6']:
                st.write("- Total payments are less than total bills")
            
            # Credit utilization
            credit_util = (input_data['BILL_AMT1'] + input_data['BILL_AMT2'] + input_data['BILL_AMT3'] + \
                          input_data['BILL_AMT4'] + input_data['BILL_AMT5'] + input_data['BILL_AMT6']) / input_data['LIMIT_BAL']
            if credit_util > 0.7:
                st.write(f"- High credit utilization ({credit_util:.0%} of limit)")
            
        else:
            st.success(f"Low Risk of Default (Probability: {probability:.2%})")

if __name__ == "__main__":
    main()
