import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go

# 1. Ρύθμιση της σελίδας
st.set_page_config(page_title="CardioPredict AI", layout="wide")

# 2. Φόρτωση του μοντέλου και του scaler
@st.cache_resource
def load_assets():
    # Φόρτωση του αποθηκευμένου μοντέλου ANN
    model = tf.keras.models.load_model('best_heart_model.keras')
    # Φόρτωση του scaler που δημιουργήθηκε στο Notebook
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Σφάλμα κατά τη φόρτωση των αρχείων: {e}")
    st.stop()

# 3. Διασύνδεση Χρήστη (UI)
st.title("🩺 Advanced Cardiac Risk Prediction System")
st.write("Σύστημα Υποστήριξης Κλινικών Αποφάσεων - Neural Network Analysis")
st.markdown("---")

# Χάρτες αντιστοίχισης για φιλική εμφάνιση
binary_map = {1: "Ναι", 0: "Όχι"}
edu_map = {1: "Δημοτικό", 2: "Γυμνάσιο", 3: "Λύκειο", 4: "Πανεπιστήμιο"}

# Δημιουργία στηλών για οργανωμένη εισαγωγή
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
    totChol = st.number_input("Χοληστερίνη (mg/dL)", 100, 600, 200)
    sysBP = st.number_input("Συστολική Πίεση (sysBP)", 80, 250, 120)
    diaBP = st.number_input("Διαστολική Πίεση (diaBP)", 40, 150, 80)
    
    # Εισαγωγή Ύψους/Βάρους για αυτόματο υπολογισμό BMI
    weight = st.number_input("Βάρος (kg)", 30.0, 250.0, 75.0)
    height_cm = st.number_input("Ύψος (cm)", 100.0, 250.0, 175.0)
    
    heartRate = st.number_input("Καρδιακοί Παλμοί", 40, 150, 75)
    glucose = st.number_input("Γλυκόζη (mg/dL)", 40, 500, 85)

# 4. Υπολογισμοί "Κάτω από το καπό"
pulse_pressure = sysBP - diaBP
# Υπολογισμός BMI: βάρος / (ύψος σε μέτρα)^2
bmi = weight / ((height_cm / 100) ** 2)

# 5. Εκτέλεση Πρόβλεψης
if st.button("🚀 Ανάλυση Κινδύνου"):
    # Δημιουργία της λίστας εισόδου με την ακριβή σειρά του Notebook
    features = [
        male, age, education, currentSmoker, cigsPerDay, BPMeds, 
        prevalentStroke, prevalentHyp, diabetes, totChol, 
        sysBP, diaBP, bmi, heartRate, glucose, pulse_pressure
    ]
    
    # Scaling των δεδομένων
    features_array = np.array([features])
    features_scaled = scaler.transform(features_array)
    
    # Πρόβλεψη πιθανότητας από το Νευρωνικό Δίκτυο
    prediction_proba = model.predict(features_scaled)[0][0]
    
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.subheader("Στατιστικά")
        st.metric("Υπολογισμένο BMI", f"{bmi:.1f}")
        st.metric("Pulse Pressure", f"{pulse_pressure:.0f} mmHg")
        
        if prediction_proba > 0.5:
            st.error(f"⚠️ ΥΨΗΛΟΣ ΚΙΝΔΥΝΟΣ: {prediction_proba*100:.1f}%")
        else:
            st.success(f"✅ ΧΑΜΗΛΟΣ ΚΙΝΔΥΝΟΣ: {prediction_proba*100:.1f}%")

    with res_col2:
        # Οπτικοποίηση με Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction_proba * 100,
            title = {'text': "Πιθανότητα Καρδιαγγειακής Νόσου (10 έτη)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps' : [
                    {'range': [0, 20], 'color': "#2ecc71"}, # Πράσινο
                    {'range': [20, 50], 'color': "#f1c40f"}, # Κίτρινο
                    {'range': [50, 100], 'color': "#e74c3c"} # Κόκκινο
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.write("Model: Artificial Neural Network (ANN)")
st.sidebar.write(f"AUC Score: 0.67")