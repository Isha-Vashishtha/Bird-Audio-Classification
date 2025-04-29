import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to make predictions
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit app
def main():
    st.title("Machine Learning Model Prediction App")
    
    st.write("Enter the input data for prediction:")
    
    # Example input fields (customize based on your model's requirements)
    input1 = st.number_input("Input 1", min_value=0.0, max_value=100.0, value=50.0)
    input2 = st.number_input("Input 2", min_value=0.0, max_value=100.0, value=50.0)
    
    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'input1': [input1],
        'input2': [input2]
    })
    
    # Load the model
    model = load_model()
    
    # Make prediction
    if st.button("Predict"):
        prediction = predict(model, input_data)
        st.write(f"Prediction: {prediction[0]}")
    
if __name__ == "__main__":
    main()
