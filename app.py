import pandas as pd
import pickle
import mimetypes
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder

# Fix for Windows ignoring CSS files
mimetypes.add_type('text/css', '.css')

app = Flask(__name__)

# --- 1. Load Model and Data ---
try:
    with open('hypertention.pkl', 'rb') as file:
        model = pickle.load(file)
    df = pd.read_csv('hypertension_dataset.csv')
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

# --- 2. Setup Encoders and Dropdown Options ---
categorical_cols = ['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    le.fit(df[col].unique())
    label_encoders[col] = le

options = {
    'bp_history_options': df['BP_History'].unique().tolist(),
    'medication_options': df['Medication'].unique().tolist(),
    'family_history_options': df['Family_History'].unique().tolist(),
    'exercise_level_options': df['Exercise_Level'].unique().tolist(),
    'smoking_status_options': df['Smoking_Status'].unique().tolist()
}

# --- 3. App Routes ---
@app.route('/')
def home():
    # Load the page with no background alerts
    return render_template('index.html', bg_status='', **options)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    processed_input = {}

    try:
        # Process Numerical Inputs
        processed_input['Age'] = int(data['Age'])
        processed_input['Salt_Intake'] = float(data['Salt_Intake'])
        processed_input['Stress_Score'] = int(data['Stress_Score'])
        processed_input['Sleep_Duration'] = float(data['Sleep_Duration'])
        processed_input['BMI'] = float(data['BMI'])

        # Process Categorical Inputs
        mapping = {
            'BP_History': 'bp_History',
            'Medication': 'medication',
            'Family_History': 'Family_History',
            'Exercise_Level': 'exercise_Level',
            'Smoking_Status': 'Smoking_Status'
        }

        for form_key, model_key in mapping.items():
            processed_input[model_key] = label_encoders[form_key].transform([data[form_key]])[0]

        # Order Features exactly as the model expects
        expected_features = [
            'Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI',
            'Family_History', 'Smoking_Status', 'bp_History', 'medication', 'exercise_Level'
        ]
        
        input_values = [processed_input[f] for f in expected_features]
        input_df = pd.DataFrame([input_values], columns=expected_features)

        # Make Prediction
        prediction = model.predict(input_df)[0]
        
        # Determine Color Theme and Text
        if prediction == 1:
            prediction_text = "Diagnostic Alert: Patient is likely to have Hypertension."
            bg_status = "danger-bg" # Triggers Red Siren
        else:
            prediction_text = "Diagnostic Clear: Patient is not likely to have Hypertension."
            bg_status = "safe-bg" # Triggers Green Siren
        
        return render_template('index.html', prediction_text=prediction_text, bg_status=bg_status, **options)

    except Exception as e:
        error_message = f"Error processing input: {str(e)}"
        return render_template('index.html', error_message=error_message, bg_status='', **options)

if __name__ == '__main__':
    app.run(debug=True)
