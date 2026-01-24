import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go

# 1. Ρύθμιση της σελίδας
st.set_page_config(page_title="CardioPredict AI", layout="centered")

# 2. Φόρτωση του μοντέλου και του scaler
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('best_heart_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Σφάλμα κατά τη φόρτωση των αρχείων: {e}")
    st.stop()

# 3. Διασύνδεση Χρήστη (UI)
st.title("🩺 Cardiac Risk Prediction System")
st.write("Σύστημα Υποστήριξης Κλινικών Αποφάσεων")
st.markdown("---")

# Χάρτες αντιστοίχισης
binary_map = {1: "Ναι", 0: "Όχι"}
edu_map = {1: "Δημοτικό", 2: "Γυμνάσιο", 3: "Λύκειο", 4: "Πανεπιστήμιο"}

# Δημιουργία στηλών
col1, col2 = st.columns(2)

with col1:
    st.subheader("Δημογραφικά & Ιστορικό")
    male = st.selectbox("Φύλο", options=[1, 0], format_func=lambda x: "Άνδρας" if x==1 else "Γυναίκα")
    age = st.number_input("Ηλικία", 18, 100, 45)
    education = st.selectbox("Επίπεδο Εκπαίδευσης", options=[1, 2, 3, 4], format_func=lambda x: edu_map[x])
    currentSmoker = st.selectbox("Καπνιστής", options=[1, 0], format_func=lambda x: binary_map[x])
    cigsPerDay = st.number_input("Τσιγάρα ανά ημέρα", 0, 100, 0)
    BPMeds = st.selectbox("Φάρμακα για πίεση", options=[1, 0], format_func=lambda x: binary_map[x])
    prevalentStroke = st.selectbox("Ιστορικό Εγκεφαλικού", options=[1, 0], format_func=lambda x: binary_map[x])
    prevalentHyp = st.selectbox("Υπέρταση", options=[1, 0], format_func=lambda x: binary_map[x])

with col2:
    st.subheader("Κλινικές Μετρήσεις")
    diabetes = st.selectbox("Διαβήτης", options=[1, 0], format_func=lambda x: binary_map[x])
    totChol = st.number_input("Χοληστερίνη (totChol)", 100, 600, 200)
    sysBP = st.number_input("Συστολική Πίεση (sysBP)", 80, 250, 120)
    diaBP = st.number_input("Διαστολική Πίεση (diaBP)", 40, 150, 80)
    BMI = st.number_input("Δείκτης Μάζας Σώματος (BMI)", 10.0, 50.0, 24.5)
    heartRate = st.number_input("Καρδιακοί Παλμοί", 40, 150, 75)
    glucose = st.number_input("Γλυκόζη", 40, 500, 85)

# 4. Υπολογισμός Pulse Pressure
pulse_pressure = sysBP - diaBP

# 5. Πρόβλεψη
if st.button("🚀 Ανάλυση Κινδύνου"):
    features = [
        male, age, education, currentSmoker, cigsPerDay, BPMeds, 
        prevalentStroke, prevalentHyp, diabetes, totChol, 
        sysBP, diaBP, BMI, heartRate, glucose, pulse_pressure
    ]
    
    features_array = np.array([features])
    features_scaled = scaler.transform(features_array)
    prediction_proba = model.predict(features_scaled)[0][0]
    
    st.markdown("---")
    st.subheader("Αποτελέσματα Πρόβλεψης")

    # Οπτικοποίηση με Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction_proba * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Πιθανότητα Καρδιαγγειακής Νόσου (10 έτη)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps' : [
                {'range': [0, 20], 'color': "#2ecc71"},
                {'range': [20, 50], 'color': "#f1c40f"},
                {'range': [50, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    st.plotly_chart(fig)

    if prediction_proba > 0.5:
        st.error(f"⚠️ ΥΨΗΛΟΣ ΚΙΝΔΥΝΟΣ: {prediction_proba*100:.1f}%")
        st.info("Σύσταση: Άμεση κλινική αξιολόγηση και ρύθμιση παραγόντων κινδύνου.")
    else:
        st.success(f"✅ ΧΑΜΗΛΟΣ ΚΙΝΔΥΝΟΣ: {prediction_proba*100:.1f}%")

st.sidebar.markdown("---")
st.sidebar.write("**Tech Stack:** ANN, TensorFlow, Streamlit")