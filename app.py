import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# 1. Ρύθμιση της σελίδας
st.set_page_config(page_title="CardioPredict AI - Pfizer Project", layout="centered")

# 2. Φόρτωση του μοντέλου και του scaler
# Βεβαιώσου ότι αυτά τα αρχεία είναι στον ίδιο φάκελο με το app.py
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
st.write("Εργαλείο υποστήριξης κλινικών αποφάσεων για την προληπτική ιατρική.")
st.info("Συμπληρώστε τα στοιχεία του ασθενούς για να λάβετε την πρόβλεψη κινδύνου σε βάθος 10ετίας.")

# Δημιουργία στηλών για πιο όμορφο layout
col1, col2 = st.columns(2)

with col1:
    male = st.selectbox("Φύλο", options=[0, 1], format_func=lambda x: "Άνδρας" if x==1 else "Γυναίκα")
    age = st.number_input("Ηλικία", 18, 100, 45)
    education = st.selectbox("Επίπεδο Εκπαίδευσης (1-4)", [1, 2, 3, 4], help="1: Δημοτικό, 4: Πανεπιστήμιο")
    currentSmoker = st.selectbox("Είναι καπνιστής;", [0, 1])
    cigsPerDay = st.number_input("Τσιγάρα ανά ημέρα", 0, 100, 0)
    BPMeds = st.selectbox("Λήψη φαρμάκων για πίεση", [0, 1])
    prevalentStroke = st.selectbox("Ιστορικό Εγκεφαλικού", [0, 1])
    prevalentHyp = st.selectbox("Υπέρταση", [0, 1])

with col2:
    diabetes = st.selectbox("Διαβήτης", [0, 1])
    totChol = st.number_input("Χοληστερίνη (totChol)", 100, 600, 200)
    sysBP = st.number_input("Συστολική Πίεση", 80, 250, 120)
    diaBP = st.number_input("Διαστολική Πίεση", 40, 150, 80)
    BMI = st.number_input("Δείκτης Μάζας Σώματος (BMI)", 10.0, 50.0, 25.0)
    heartRate = st.number_input("Καρδιακοί Παλμοί", 40, 150, 75)
    glucose = st.number_input("Γλυκόζη", 40, 500, 90)

# 4. Υπολογισμός Pulse Pressure (Το 16ο χαρακτηριστικό)
pulse_pressure = sysBP - diaBP

# 5. Πρόβλεψη
if st.button("🚀 Υπολογισμός Κινδύνου"):
    # Δημιουργία της λίστας με την ΑΚΡΙΒΗ σειρά που εκπαιδεύτηκε ο scaler
    features = [
        male, age, education, currentSmoker, cigsPerDay, BPMeds, 
        prevalentStroke, prevalentHyp, diabetes, totChol, 
        sysBP, diaBP, BMI, heartRate, glucose, pulse_pressure
    ]
    
    # Μετατροπή σε array και scaling
    features_array = np.array([features])
    features_scaled = scaler.transform(features_array)
    
    # Εκτέλεση πρόβλεψης
    prediction_proba = model.predict(features_scaled)[0][0]
    
    st.divider()
    
    # Εμφάνιση αποτελεσμάτων
    if prediction_proba > 0.5:
        st.error(f"### Υψηλός Κίνδυνος: {prediction_proba*100:.1f}%")
        st.write("Συνίσταται περαιτέρω κλινικός έλεγχος και παρακολούθηση.")
    else:
        st.success(f"### Χαμηλός Κίνδυνος: {prediction_proba*100:.1f}%")
        st.write("Ο ασθενής βρίσκεται εντός των φυσιολογικών ορίων πρόγνωσης.")

st.sidebar.markdown("---")
st.sidebar.write("Developed for **Pfizer Digital Hub Portfolio**")
st.sidebar.write("Model: Deep Neural Network (ANN)")