import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("xgboost_model.pkl")

st.title("YouTube Views Prediction")

# Input fields for features
features = []
feature_names = [
    "Red Views", "Comments", "Likes", "Dislikes", "Videos Added To Playlists",
    "Shares", "Estimated Minutes Watched", "Estimated Red Minutes Watched",
    "Average View Duration", "Average View Percentage",
    "Card Teaser Impressions", "Subscribers Gained", "Month"
]

for name in feature_names:
    value = st.number_input(name, value=0)
    features.append(value)

if st.button("Predict Views"):
    prediction = model.predict(np.array(features).reshape(1, -1))[0]
    st.success(f"Predicted Views: {prediction:.2f}")
