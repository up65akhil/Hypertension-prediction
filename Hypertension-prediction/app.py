import pandas as pd
import pickle
from flask import Flask, request, render_template # Import render_template

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
# These are the column names from your CSV and HTML form
categorical_cols_from_csv = ['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']
label_encoders = {} # Dictionary to store LabelEncoder objects

# Fit LabelEncoders for each categorical column using the full dataset's unique values
from sklearn.preprocessing import LabelEncoder # Moved import here as it's only used for fitting
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


# --- Flask Routes ---
@app.route('/')
def home():
    # Renders the initial form from the 'templates' folder
    return render_template(
        'index.html',
        bp_history_options=bp_history_options,
        medication_options=medication_options,
        family_history_options=family_history_options,
        exercise_level_options=exercise_level_options,
        smoking_status_options=smoking_status_options
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    # Dictionary to hold processed features, using the model's expected names as keys
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
        return render_template(
            'index.html',
            error_message=error_message,
            bp_history_options=bp_history_options,
            medication_options=medication_options,
            family_history_options=family_history_options,
            exercise_level_options=exercise_level_options,
            smoking_status_options=smoking_status_options
        )


    # 2. Handle Categorical Features using LabelEncoder
    # Map form input names (from CSV) to model's expected feature names (from .pkl snippet)
    form_to_model_feature_map = {
        'BP_History': 'bp_History',  # Model expects 'bp_History'
        'Medication': 'medication',  # Model expects 'medication'
        'Family_History': 'Family_History', # Name is consistent
        'Exercise_Level': 'exercise_Level', # Model expects 'exercise_Level'
        'Smoking_Status': 'Smoking_Status' # Name is consistent
    }

    for form_col, model_col in form_to_model_feature_map.items():
        try:
            # Transform the user's selected category using the pre-fitted encoder
            processed_input[model_col] = label_encoders[form_col].transform([data[form_col]])[0]
        except ValueError as e:
            # If an unseen category is encountered by LabelEncoder
            error_message = f"Error: The selected option for '{form_col}' ('{data.get(form_col, 'N/A')}') is not recognized. Please choose from the dropdowns."
            print(f"DEBUG: Error encoding categorical feature '{form_col}' with value '{data.get(form_col, 'N/A')}': {e}")
            return render_template(
                'index.html',
                error_message=error_message, # Pass error to HTML template
                bp_history_options=bp_history_options,
                medication_options=medication_options,
                family_history_options=family_history_options,
                exercise_level_options=exercise_level_options,
                smoking_status_options=smoking_status_options
            )
        except KeyError as e:
            # If a form field name is unexpected (e.g., not present in request.form)
            error_message = f"Error: Missing input for a required field. Please ensure all fields are selected/filled. (Missing: '{e}')"
            print(f"DEBUG: KeyError for form field '{e}': Not found in request.form.")
            return render_template(
                'index.html',
                error_message=error_message,
                bp_history_options=bp_history_options,
                medication_options=medication_options,
                family_history_options=family_history_options,
                exercise_level_options=exercise_level_options,
                smoking_status_options=smoking_status_options
            )


    # 3. Get the exact feature names and order from the loaded model
    # This is the most reliable way to ensure correct column order.
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
    else:
        # Fallback if model.feature_names_in_ is not available (less robust).
        # This list MUST EXACTLY match the order of features used during model training.
        # Based on hypertention.pkl snippet:
        expected_features = [
            'Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI',
            'Family_History', 'Smoking_Status', 'bp_History', 'medication', 'exercise_Level'
        ]

    # Create a list of values in the correct order based on expected_features
    # This ensures the DataFrame columns are in the precise order the model expects.
    try:
        input_values_ordered = [processed_input[feature] for feature in expected_features]
        input_df = pd.DataFrame([input_values_ordered], columns=expected_features)
    except KeyError as e:
        error_message = f"Internal Error: A required model feature ('{e}') was not correctly prepared. Please contact support."
        print(f"DEBUG: KeyError when creating input_df: Missing feature '{e}' in processed_input.")
        return render_template(
            'index.html',
            error_message=error_message,
            bp_history_options=bp_history_options,
            medication_options=medication_options,
            family_history_options=family_history_options,
            exercise_level_options=exercise_level_options,
            smoking_status_options=smoking_status_options
        )

    # --- Debugging Output (Check your terminal where Flask is running) ---
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
        return render_template(
            'index.html',
            error_message=error_message,
            bp_history_options=bp_history_options,
            medication_options=medication_options,
            family_history_options=family_history_options,
            exercise_level_options=exercise_level_options,
            smoking_status_options=smoking_status_options
        )

    # Determine prediction text
    # Assuming 1 for 'Yes' (has hypertension) and 0 for 'No' (does not have hypertension)
    if prediction == 1:
        prediction_text = "Prediction: Yes, the person is likely to have Hypertension."
    else:
        prediction_text = "Prediction: No, the person is not likely to have Hypertension."

    # Re-render the same template with the prediction result
    return render_template(
        'index.html',
        prediction_text=prediction_text, # This variable makes the result visible
        bp_history_options=bp_history_options,
        medication_options=medication_options,
        family_history_options=family_history_options,
        exercise_level_options=exercise_level_options,
        smoking_status_options=smoking_status_options
    )

if __name__ == '__main__':
    app.run(debug=True)