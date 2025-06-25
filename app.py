import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Predictor")

st.markdown("Enter passenger details below:")

# Form inputs
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["Male", "Female"])
Sex = 0 if Sex == "Male" else 1

Ageband = st.selectbox("Age Band", [0, 1, 2, 3, 4], format_func=lambda x: ["0-16", "17-32", "33-48", "49-64", "65-80"][x])
Fareband = st.selectbox("Fare Band", [0, 1, 2, 3])
Embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])
Embarked = {'S': 0, 'C': 1, 'Q': 2}[Embarked]

faimily_size = st.slider("Family Size", 1, 10, 1)
IsAlone = 1 if faimily_size == 1 else 0

Title = st.selectbox("Title", ['Mr', 'Miss', 'Mrs', 'Master', 'Rare'])
Title = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}[Title]

# Predict button
if st.button("Predict"):
    input_data = np.array([[Pclass, Sex, Ageband, Fareband, Embarked,
                            faimily_size, IsAlone, Title]])
    prediction = model.predict(input_data)[0]
    result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
    st.subheader("Prediction:")
    st.success(result)
