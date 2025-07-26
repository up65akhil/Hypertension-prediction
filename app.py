import pandas as pd
import pickle
from flask import Flask, request, render_template_string # Using render_template_string
from sklearn.preprocessing import LabelEncoder

# --- Model and Data Loading ---
try:
    with open('hypertention.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: hypertention.pkl not found. Please ensure the model file is in the same directory.")
    exit()
except Exception as e:
    print(f"Error loading model hypertention.pkl: {e}")
    exit()

try:
    df = pd.read_csv('hypertension_dataset.csv')
except FileNotFoundError:
    print("Error: hypertension_dataset.csv not found. Please ensure the dataset file is in the same directory.")
    exit()
except Exception as e:
    print(f"Error loading dataset hypertension_dataset.csv: {e}")
    exit()

app = Flask(__name__)

# --- Categorical Features and Label Encoders ---
categorical_cols_from_csv = ['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']
label_encoders = {} # Dictionary to store LabelEncoder objects

for col in categorical_cols_from_csv:
    le = LabelEncoder()
    le.fit(df[col].unique())
    label_encoders[col] = le

# Get unique values for dropdowns (for HTML template)
bp_history_options = df['BP_History'].unique().tolist()
medication_options = df['Medication'].unique().tolist()
family_history_options = df['Family_History'].unique().tolist()
exercise_level_options = df['Exercise_Level'].unique().tolist()
smoking_status_options = df['Smoking_Status'].unique().tolist()

# --- HTML Template with Embedded CSS ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hypertension Prediction</title>
    <style>
        /* Embedded CSS from static/style.css */
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #0d0d0d;
            color: #e0e0e0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            flex-direction: column;
            padding: 10px;
            box-sizing: border-box;
            overflow-y: auto;
        }

        .container {
            background-color: #1c1c1c;
            padding: 20px 40px;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
            width: 100%;
            max-width: 550px;
            box-sizing: border-box;
            border: 1px solid #2a2a2a;
            flex-shrink: 0;
        }

        h2 {
            text-align: center;
            color: #ffffff;
            margin-bottom: 20px;
            font-size: 2em;
            font-weight: 600;
        }

        .form-group {
            margin-bottom: 15px;
        }

        label {
            display: block;
            margin-bottom: 5px;
            color: #c0c0c0;
            font-weight: bold;
            font-size: 0.9em;
        }

        input[type="number"],
        select {
            width: 100%;
            padding: 10px 15px;
            border: 1px solid #3a3a3a;
            border-radius: 6px;
            box-sizing: border-box;
            font-size: 0.95em;
            background-color: #2a2a2a;
            color: #ffffff;
            outline: none;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }

        input[type="number"]:focus,
        select:focus {
            border-color: #7b68ee;
            box-shadow: 0 0 0 3px rgba(123, 104, 238, 0.3);
        }

        select {
            -webkit-appearance: none;
            -moz-appearance: none;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23e0e0e0'%3E%3Cpath d='M7 10l5 5 5-5z'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 10px center;
            background-size: 1em;
            padding-right: 30px;
        }

        button {
            width: 100%;
            padding: 12px;
            background-color: #7b68ee;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.2s ease;
            margin-top: 10px;
        }

        button:hover {
            background-color: #6a5acd;
            transform: translateY(-2px);
        }

        .prediction-highlight {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            background-color: hsl(35, 100%, 68%);
            color: #333;
            border: 1px solid hsl(35, 100%, 58%);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }

        .error-message {
            margin-top: 15px;
            padding: 12px;
            border-radius: 5px;
            text-align: center;
            font-size: 15px;
            font-weight: bold;
            background-color: #dc3545;
            color: #ffffff;
            border: 1px solid #c82333;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Hypertension Prediction</h2>
        <form action="/predict" method="post">
            <div class="form-group">
                <label for="age">Age:</label>
                <input type="number" id="age" name="Age" step="1" required>
            </div>
            <div class="form-group">
                <label for="salt_intake">Salt Intake (g/day):</label>
                <input type="number" id="salt_intake" name="Salt_Intake" step="0.1" required>
            </div>
            <div class="form-group">
                <label for="stress_score">Stress Score (0-10):</label>
                <input type="number" id="stress_score" name="Stress_Score" step="1" min="0" max="10" required>
            </div>
            <div class="form-group">
                <label for="bp_history">BP History:</label>
                <select id="bp_history" name="BP_History" required>
                    {% for option in bp_history_options %}
                    <option value="{{ option }}">{{ option }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="form-group">
                <label for="sleep_duration">Sleep Duration (hours):</label>
                <input type="number" id="sleep_duration" name="Sleep_Duration" step="0.1" required>
            </div>
            <div class="form-group">
                <label for="bmi">BMI:</label>
                <input type="number" id="bmi" name="BMI" step="0.1" required>
            </div>
            <div class="form-group">
                <label for="medication">Medication:</label>
                <select id="medication" name="Medication" required>
                    {% for option in medication_options %}
                    <option value="{{ option }}">{{ option }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="form-group">
                <label for="family_history">Family History:</label>
                <select id="family_history" name="Family_History" required>
                    {% for option in family_history_options %}
                    <option value="{{ option }}">{{ option }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="form-group">
                <label for="exercise_level">Exercise Level:</label>
                <select id="exercise_level" name="Exercise_Level" required>
                    {% for option in exercise_level_options %}
                    <option value="{{ option }}">{{ option }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="form-group">
                <label for="smoking_status">Smoking Status:</label>
                <select id="smoking_status" name="Smoking_Status" required>
                    {% for option in smoking_status_options %}
                    <option value="{{ option }}">{{ option }}</option>
                    {% endfor %}
                </select>
            </div>
            <button type="submit">Predict</button>
        </form>
    </div>

    {% if error_message %}
    <div class="error-message">
        {{ error_message }}
    </div>
    {% endif %}

    {% if prediction_text %}
    <div class="prediction-highlight">
        {{ prediction_text }}
    </div>
    {% endif %}
</body>
</html>
"""

# --- Flask Routes ---
@app.route('/')
def home():
    # Renders the initial form from the HTML_TEMPLATE string
    return render_template_string(
        HTML_TEMPLATE,
        bp_history_options=bp_history_options,
        medication_options=medication_options,
        family_history_options=family_history_options,
        exercise_level_options=exercise_level_options,
        smoking_status_options=smoking_status_options
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    processed_input = {}

    # 1. Handle Numerical Features (names are consistent)
    try:
        processed_input['Age'] = int(data['Age'])
        processed_input['Salt_Intake'] = float(data['Salt_Intake'])
        processed_input['Stress_Score'] = int(data['Stress_Score'])
        processed_input['Sleep_Duration'] = float(data['Sleep_Duration'])
        processed_input['BMI'] = float(data['BMI'])
    except ValueError as e:
        error_message = f"Error: Please ensure all numerical fields are filled correctly. Details: {e}"
        print(f"DEBUG: Numerical input conversion error: {e}")
        return render_template_string(
            HTML_TEMPLATE,
            error_message=error_message,
            bp_history_options=bp_history_options,
            medication_options=medication_options,
            family_history_options=family_history_options,
            exercise_level_options=exercise_level_options,
            smoking_status_options=smoking_status_options
        )

    # 2. Handle Categorical Features using LabelEncoder
    form_to_model_feature_map = {
        'BP_History': 'bp_History',
        'Medication': 'medication',
        'Family_History': 'Family_History',
        'Exercise_Level': 'exercise_Level',
        'Smoking_Status': 'Smoking_Status'
    }

    for form_col, model_col in form_to_model_feature_map.items():
        try:
            processed_input[model_col] = label_encoders[form_col].transform([data[form_col]])[0]
        except ValueError as e:
            error_message = f"Error: The selected option for '{form_col}' ('{data.get(form_col, 'N/A')}') is not recognized. Please choose from the dropdowns."
            print(f"DEBUG: Error encoding categorical feature '{form_col}' with value '{data.get(form_col, 'N/A')}': {e}")
            return render_template_string(
                HTML_TEMPLATE,
                error_message=error_message,
                bp_history_options=bp_history_options,
                medication_options=medication_options,
                family_history_options=family_history_options,
                exercise_level_options=exercise_level_options,
                smoking_status_options=smoking_status_options
            )
        except KeyError as e:
            error_message = f"Error: Missing input for a required field. Please ensure all fields are selected/filled. (Missing: '{e}')"
            print(f"DEBUG: KeyError for form field '{e}': Not found in request.form.")
            return render_template_string(
                HTML_TEMPLATE,
                error_message=error_message,
                bp_history_options=bp_history_options,
                medication_options=medication_options,
                family_history_options=family_history_options,
                exercise_level_options=exercise_level_options,
                smoking_status_options=smoking_status_options
            )

    # 3. Get the exact feature names and order from the loaded model
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
    else:
        expected_features = [
            'Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI',
            'Family_History', 'Smoking_Status', 'bp_History', 'medication', 'exercise_Level'
        ]

    try:
        input_values_ordered = [processed_input[feature] for feature in expected_features]
        input_df = pd.DataFrame([input_values_ordered], columns=expected_features)
    except KeyError as e:
        error_message = f"Internal Error: A required model feature ('{e}') was not correctly prepared. Please contact support."
        print(f"DEBUG: KeyError when creating input_df: Missing feature '{e}' in processed_input.")
        return render_template_string(
            HTML_TEMPLATE,
            error_message=error_message,
            bp_history_options=bp_history_options,
            medication_options=medication_options,
            family_history_options=family_history_options,
            exercise_level_options=exercise_level_options,
            smoking_status_options=smoking_status_options
        )

    print("\n--- Debugging Input Data ---")
    print("Processed Input Dictionary (for model):", processed_input)
    print("Model's Expected Features Order:", expected_features)
    print("DataFrame for Prediction:\n", input_df)
    print("---------------------------\n")

    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        error_message = f"Error during model prediction: {e}. Ensure your scikit-learn version matches '1.6.1' or a compatible version."
        print(f"DEBUG: Model prediction error: {e}")
        return render_template_string(
            HTML_TEMPLATE,
            error_message=error_message,
            bp_history_options=bp_history_options,
            medication_options=medication_options,
            family_history_options=family_history_options,
            exercise_level_options=exercise_level_options,
            smoking_status_options=smoking_status_options
        )

    if prediction == 1:
        prediction_text = "Prediction: Yes, the person is likely to have Hypertension."
    else:
        prediction_text = "Prediction: No, the person is not likely to have Hypertension."

    return render_template_string(
        HTML_TEMPLATE,
        prediction_text=prediction_text,
        bp_history_options=bp_history_options,
        medication_options=medication_options,
        family_history_options=family_history_options,
        exercise_level_options=exercise_level_options,
        smoking_status_options=smoking_status_options
    )

if __name__ == '__main__':
    app.run(debug=True)