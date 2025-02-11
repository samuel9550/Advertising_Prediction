import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("linear_regression_model.pkl")

# Streamlit UI
st.title("Advertising Budget & Sales Prediction")
st.write("Enter the advertising budgets to predict the expected sales.")

# User input
tv_budget = st.number_input("TV Budget ($)", min_value=0.0, value=100.0, step=10.0)
radio_budget = st.number_input("Radio Budget ($)", min_value=0.0, value=50.0, step=5.0)
newspaper_budget = st.number_input("Newspaper Budget ($)", min_value=0.0, value=30.0, step=5.0)

# Prediction
if st.button("Predict Sales"):
    input_data = pd.DataFrame([[tv_budget, radio_budget, newspaper_budget]], 
                              columns=['TV_Budget', 'Radio_Budget', 'Newspaper_Budget'])
    predicted_sales = model.predict(input_data)[0]
    st.success(f"Estimated Sales: ${predicted_sales:.2f}")


