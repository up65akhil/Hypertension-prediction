import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Hypertension Diagnostic", page_icon="❤️", layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 0rem !important;
            max-width: 95% !important;
        }
        
        div[data-testid="stForm"] {
            padding: 1rem !important;
        }

        .heart-icon {
            color: #dc3545;
            font-size: 2.5rem;
            display: inline-block;
            animation: heartbeat 1.5s infinite;
            vertical-align: middle;
            margin-right: 15px;
        }

        @keyframes heartbeat {
            0% { transform: scale(1); }
            15% { transform: scale(1.15); }
            30% { transform: scale(1); }
            45% { transform: scale(1.15); }
            60% { transform: scale(1); }
        }

        .alert-overlay {
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            pointer-events: none;
            z-index: 9999;
        }
        .blink-red { animation: redSiren 1.5s infinite ease-in-out; }
        .blink-green { animation: greenSiren 1.5s infinite ease-in-out; }
        
        @keyframes redSiren {
            0% { box-shadow: inset 0 0 0px 0px rgba(255, 0, 0, 0); }
            50% { box-shadow: inset 0 0 120px 40px rgba(220, 53, 69, 0.9); }
            100% { box-shadow: inset 0 0 0px 0px rgba(255, 0, 0, 0); }
        }
        @keyframes greenSiren {
            0% { box-shadow: inset 0 0 0px 0px rgba(0, 255, 0, 0); }
            50% { box-shadow: inset 0 0 120px 40px rgba(40, 167, 69, 0.9); }
            100% { box-shadow: inset 0 0 0px 0px rgba(0, 255, 0, 0); }
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        with open('hypertention.pkl', 'rb') as file:
            model = pickle.load(file)
        df = pd.read_csv('hypertension_dataset.csv')
        return model, df
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

model, df = load_assets()

@st.cache_data
def setup_encoders(_dataframe):
    categorical_cols = ['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']
    label_encoders = {}
    options = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(_dataframe[col].unique())
        label_encoders[col] = le
        options[col] = _dataframe[col].unique().tolist()
    return label_encoders, options

label_encoders, options = setup_encoders(df)

with st.sidebar:
    st.title("About This Tool")
    st.divider()
    
    st.subheader("🤖 Project Details")
    st.markdown("""
    This diagnostic tool is powered by a machine learning model trained to identify hypertension risks based on patient vitals and lifestyle factors.
    
    * **Algorithm:** Random Forest Classifier
    * **Dataset Source:** Kaggle
    * **Model Accuracy:** 89%
    """)
    
    st.divider()
    
    st.subheader("⚠️ Clinical Disclaimer")
    st.warning("This tool provides a preliminary AI-assisted risk assessment. It does not substitute professional medical diagnosis.")
    st.info("Ensure all patient vitals are accurately measured before running the diagnostic analysis.")

st.markdown("<div><span class='heart-icon'>❤️</span><h2 style='display:inline; vertical-align: middle;'>Hypertension Risk Assessment</h2></div>", unsafe_allow_html=True)
st.write("Fill in the patient's vitals below. This compact view is designed to evaluate risk without leaving the screen.")

with st.form("diagnostic_form"):
    st.markdown("**Biometrics & Lifestyle**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: age = st.number_input("Age", min_value=0, step=1, value=30)
    with c2: bmi = st.number_input("BMI (kg/m²)", min_value=0.0, step=0.1, value=22.5)
    with c3: sleep_duration = st.number_input("Sleep (Hrs)", min_value=0.0, step=0.1, value=7.0)
    with c4: salt_intake = st.number_input("Salt (g/day)", min_value=0.0, step=0.1, value=5.0)
    with c5: stress_score = st.number_input("Stress (0-10)", min_value=0, max_value=10, step=1, value=5)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**Medical History**")
    c6, c7, c8, c9, c10 = st.columns(5)
    with c6: bp_history = st.selectbox("BP History", options['BP_History'])
    with c7: medication = st.selectbox("Medication", options['Medication'])
    with c8: family_history = st.selectbox("Family History", options['Family_History'])
    with c9: exercise_level = st.selectbox("Exercise", options['Exercise_Level'])
    with c10: smoking_status = st.selectbox("Smoking", options['Smoking_Status'])

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="RUN DIAGNOSTIC ANALYSIS", use_container_width=True)

if submit_button:
    try:
        processed_bp_history = label_encoders['BP_History'].transform([bp_history])[0]
        processed_medication = label_encoders['Medication'].transform([medication])[0]
        processed_family_history = label_encoders['Family_History'].transform([family_history])[0]
        processed_exercise_level = label_encoders['Exercise_Level'].transform([exercise_level])[0]
        processed_smoking_status = label_encoders['Smoking_Status'].transform([smoking_status])[0]

        expected_features = [
            'Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI',
            'Family_History', 'Smoking_Status', 'bp_History', 'medication', 'exercise_Level'
        ]
        
        input_values = [
            age, salt_intake, stress_score, sleep_duration, bmi,
            processed_family_history, processed_smoking_status, processed_bp_history, 
            processed_medication, processed_exercise_level
        ]
        
        input_df = pd.DataFrame([input_values], columns=expected_features)
        prediction = model.predict(input_df)[0]
        
        if prediction == 1:
            st.markdown('<div class="alert-overlay blink-red"></div>', unsafe_allow_html=True)
            st.error("🚨 **Diagnostic Alert:** Patient is highly likely to have Hypertension. Recommend immediate clinical review.")
        else:
            st.markdown('<div class="alert-overlay blink-green"></div>', unsafe_allow_html=True)
            st.success("✅ **Diagnostic Clear:** Patient is not likely to have Hypertension. Continue standard care.")

    except Exception as e:
        st.error(f"Error processing input: {str(e)}")
