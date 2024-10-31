# Step 1: Import dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import gradio as gr

# Step 2: Load and prepare the dataset
# Load the data
heart_data = pd.read_csv("heart_disease.csv")
# Split the data into features (X) and target (Y)
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Step 3: Train the Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)

# Step 4: Define the prediction function without external model querying
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Check for empty inputs
    if any(v is None or v == "" for v in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]):
        return "Please enter valid entries."

    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    try:
        # Make the prediction
        prediction = model.predict(input_data_as_numpy_array)

        # Interpret prediction
        result = 'The Person has Heart Disease' if prediction[0] == 1 else 'The Person does not have Heart Disease'

        # Display result
        return result

    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

# Step 5: Define the Gradio interface
inputs = [
    gr.Number(label="Age *"),
    gr.Number(label="Sex (0 = female, 1 = male) *"),
    gr.Number(label="Chest Pain Type (0-3) *"),
    gr.Number(label="Resting Blood Pressure (>80 & <200) *"),
    gr.Number(label="Cholesterol (>120) *"),
    gr.Number(label="Fasting Blood Sugar > 120 mg/dl (0 = false, 1 = true) *"),
    gr.Number(label="Resting ECG Results (0-2) *"),
    gr.Number(label="Maximum Heart Rate Achieved (<200) *"),
    gr.Number(label="Exercise Induced Angina (0 = no, 1 = yes) *"),
    gr.Number(label="ST Depression Induced by Exercise (0.0-4.5) *"),
    gr.Number(label="Slope of Peak Exercise ST Segment (0-2) *"),
    gr.Number(label="Number of Major Vessels Colored by Fluoroscopy (0-3) *"),
    gr.Number(label="Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect) *")
]

output = gr.Textbox(label="Prediction Result", interactive=True)

# CSS for the background and design
css = """
    body {
        background-color: lightblue; /* Fallback color */
        background-image: url('/content/Heart.jpg'); /* Path to your uploaded image */
        background-size: cover; /* Cover the entire area */
        background-position: center; /* Center the background */
        background-repeat: no-repeat; /* Prevent the background from repeating */
    }
    .gradio-container {
        font-family: Arial, sans-serif;
        color: #333;
    }
    .gr-button {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .gr-button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    input[required]:invalid {
        border: 2px solid red; /* Red border for required fields */
    }
    input[required]:valid {
        border: 2px solid green; /* Green border for valid fields */
    }
"""
iface = gr.Interface(
    fn=predict_heart_disease,
    inputs=inputs,
    outputs=output,
    title="<h1 style='font-size: 28px; color: green;'>Heart Disease Prediction System</h1>",
    description="<p style='font-size: 18px; color: black;'>Enter patient details to predict heart disease <span style='color: red;'>*</span></p>",
    css=css,
    allow_flagging="manual",  # Allows flagging
    flagging_options=["Save"],  
)


iface.launch()
