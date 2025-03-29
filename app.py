import streamlit as st
import joblib
import pandas as pd
import io
from PIL import Image

# Load the model
model = joblib.load('credit_card_default_model.pkl')

def predict_default(input_data):
    """Predict default for a single record"""
    # Create features from input data
    input_df = pd.DataFrame([input_data])
    
    return predict_default_batch(input_df)

def predict_default_batch(input_df):
    """Predict default for multiple records in a DataFrame"""
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
    
    # Make predictions
    predictions = model.predict(input_df[features])
    probabilities = model.predict_proba(input_df[features])[:, 1]
    
    # Add predictions to the DataFrame
    input_df['DEFAULT_PREDICTION'] = predictions
    input_df['DEFAULT_PROBABILITY'] = probabilities
    
    return input_df

def main():
    st.title("Credit Card Default Risk Predictor")
    st.write("Predict the likelihood of credit card holders defaulting on their next payment.")
    
    # Add sidebar with options
    st.sidebar.header("Prediction Mode")
    prediction_mode = st.sidebar.radio("Select prediction mode:", 
                                     ("Single Prediction", "Batch Prediction"))
    
    if prediction_mode == "Single Prediction":
        # Single prediction form
        with st.form("single_prediction_form"):
            st.header("Customer Information")
            col1, col2 = st.columns(2)
            with col1:
                limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0)
            with col2:
                age = st.number_input("Age (AGE)", min_value=18, max_value=100)
            
            st.header("Payment History (Delays in Months)")
            cols = st.columns(6)
            with cols[0]:
                pay_0 = st.number_input("September (PAY_0)", min_value=-2, max_value=8)
            with cols[1]:
                pay_2 = st.number_input("August (PAY_2)", min_value=-2, max_value=8)
            with cols[2]:
                pay_3 = st.number_input("July (PAY_3)", min_value=-2, max_value=8)
            with cols[3]:
                pay_4 = st.number_input("June (PAY_4)", min_value=-2, max_value=8)
            with cols[4]:
                pay_5 = st.number_input("May (PAY_5)", min_value=-2, max_value=8)
            with cols[5]:
                pay_6 = st.number_input("April (PAY_6)", min_value=-2, max_value=8)
            
            st.header("Payment Amounts")
            cols = st.columns(6)
            with cols[0]:
                pay_amt1 = st.number_input("September (PAY_AMT1)", min_value=0)
            with cols[1]:
                pay_amt2 = st.number_input("August (PAY_AMT2)", min_value=0)
            with cols[2]:
                pay_amt3 = st.number_input("July (PAY_AMT3)", min_value=0)
            with cols[3]:
                pay_amt4 = st.number_input("June (PAY_AMT4)", min_value=0)
            with cols[4]:
                pay_amt5 = st.number_input("May (PAY_AMT5)", min_value=0)
            with cols[5]:
                pay_amt6 = st.number_input("April (PAY_AMT6)", min_value=0)
            
            st.header("Bill Statements")
            cols = st.columns(6)
            with cols[0]:
                bill_amt1 = st.number_input("September (BILL_AMT1)", min_value=0)
            with cols[1]:
                bill_amt2 = st.number_input("August (BILL_AMT2)", min_value=0)
            with cols[2]:
                bill_amt3 = st.number_input("July (BILL_AMT3)", min_value=0)
            with cols[3]:
                bill_amt4 = st.number_input("June (BILL_AMT4)", min_value=0)
            with cols[4]:
                bill_amt5 = st.number_input("May (BILL_AMT5)", min_value=0)
            with cols[5]:
                bill_amt6 = st.number_input("April (BILL_AMT6)", min_value=0)
            
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
            result_df = predict_default(input_data)
            prediction = result_df['DEFAULT_PREDICTION'].iloc[0]
            probability = result_df['DEFAULT_PROBABILITY'].iloc[0]
            
            # Display results
            st.subheader("Prediction Results")
            if prediction == 1:
                st.error(f"High Risk of Default (Probability: {probability:.2%})")
                
                # Explain key risk factors
                st.write("Key Risk Factors Identified:")
                
                # Payment delays
                avg_delay = result_df['AVG_PAY_DELAY'].iloc[0]
                if avg_delay > 0:
                    st.write(f"- Average payment delay of {avg_delay:.1f} months")
                
                # Payment to bill ratio
                if result_df['PAYMENT_TO_BILL_RATIO'].iloc[0] < 1:
                    st.write("- Total payments are less than total bills")
                
                # Credit utilization
                credit_util = result_df['CREDIT_UTILIZATION'].iloc[0]
                if credit_util > 0.7:
                    st.write(f"- High credit utilization ({credit_util:.0%} of limit)")
                
            else:
                st.success(f"Low Risk of Default (Probability: {probability:.2%})")
    
    else:
        # Batch prediction
        st.header("Batch Prediction")
        st.write("Upload a CSV file containing customer data for batch prediction.")
        
        # Download template
        st.markdown("### Download Template")
        st.write("Use this template to ensure your file has the correct format:")
        
        # Create template DataFrame
        template_data = {
            'LIMIT_BAL': [20000],
            'AGE': [30],
            'PAY_0': [0],
            'PAY_2': [0],
            'PAY_3': [0],
            'PAY_4': [0],
            'PAY_5': [0],
            'PAY_6': [0],
            'PAY_AMT1': [1000],
            'PAY_AMT2': [1000],
            'PAY_AMT3': [1000],
            'PAY_AMT4': [1000],
            'PAY_AMT5': [1000],
            'PAY_AMT6': [1000],
            'BILL_AMT1': [2000],
            'BILL_AMT2': [2000],
            'BILL_AMT3': [2000],
            'BILL_AMT4': [2000],
            'BILL_AMT5': [2000],
            'BILL_AMT6': [2000]
        }
        template_df = pd.DataFrame(template_data)
        
        # Convert to CSV for download
        csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Template CSV",
            data=csv,
            file_name='credit_card_default_template.csv',
            mime='text/csv'
        )
        
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                input_df = pd.read_csv(uploaded_file)
                
                # Show a preview
                st.subheader("Uploaded Data Preview")
                st.write(input_df.head())
                
                # Validate columns
                required_columns = [
                    'LIMIT_BAL', 'AGE', 
                    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
                    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'
                ]
                
                missing_cols = [col for col in required_columns if col not in input_df.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                else:
                    if st.button("Run Batch Prediction"):
                        with st.spinner('Processing...'):
                            # Make predictions
                            result_df = predict_default_batch(input_df)
                            
                            # Show results
                            st.subheader("Prediction Results")
                            
                            # Summary statistics
                            st.write(f"Total records processed: {len(result_df)}")
                            st.write(f"High risk predictions: {result_df['DEFAULT_PREDICTION'].sum()} "
                                    f"({result_df['DEFAULT_PREDICTION'].mean():.1%})")
                            
                            # Show high risk customers first
                            st.write("### High Risk Customers")
                            high_risk_df = result_df[result_df['DEFAULT_PREDICTION'] == 1].sort_values(
                                'DEFAULT_PROBABILITY', ascending=False)
                            st.dataframe(high_risk_df)
                            
                            # Download results
                            st.markdown("### Download Results")
                            result_csv = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=result_csv,
                                file_name='credit_card_default_predictions.csv',
                                mime='text/csv'
                            )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
