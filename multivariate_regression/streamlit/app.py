import streamlit as st
import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("House Price Prediction App")

x1 = st.number_input("Enter Area (sqft)")
x2 = st.number_input("Enter Bedrooms")
x3 = st.number_input("Enter Age (years)")

if st.button("Predict"):
    features = np.array([[x1, x2, x3]])
    prediction = model.predict(features)
    st.success(f"Predicted Price: {prediction[0]:.2f}")
