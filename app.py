import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler



# Load the model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('breast_cancer_model.pkl')
        scaler = joblib.load('scaler (1).pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading the model or scaler: {str(e)}")
        return None, None

model, scaler = load_model_and_scaler()

# Function to map user inputs to dataset features
def map_answers_to_features(answers):
    return [
        answers['tumor_size'], answers['tumor_texture'],
        answers['tumor_size'] * 2 * np.pi, np.pi * answers['tumor_size']**2,
        answers['tumor_smoothness'], answers['tumor_density'],
        answers['tumor_irregularity'], answers['tumor_irregularity'] * 0.5,
        answers['tumor_symmetry'], answers['fractal_dimension'],
        answers['radius_error'], answers['texture_error'],
        answers['perimeter_error'], answers['area_error'],
        answers['smoothness_error'], answers['compactness_error'],
        answers['concavity_error'], answers['concave_points_error'],
        answers['symmetry_error'], answers['fractal_dimension_error'],
        answers['tumor_size'] + answers['radius_error'],
        answers['tumor_texture'] + answers['texture_error'],
        (answers['tumor_size'] + answers['radius_error']) * 2 * np.pi,
        np.pi * (answers['tumor_size'] + answers['radius_error'])**2,
        answers['tumor_smoothness'] + answers['smoothness_error'],
        answers['tumor_density'] + answers['compactness_error'],
        answers['tumor_irregularity'] + answers['concavity_error'],
        (answers['tumor_irregularity'] + answers['concavity_error']) * 0.5,
        answers['tumor_symmetry'] + answers['symmetry_error'],
        answers['fractal_dimension'] + answers['fractal_dimension_error']
    ]

# Function to make predictions
# Function to make predictions
def predict_tumor(user_answers):
    if model is None or scaler is None:
        return "Model or scaler not loaded. Unable to make predictions.", 0

    try:
        input_data = map_answers_to_features(user_answers)
        input_data_array = np.asarray(input_data).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data_array)
        
        # Get prediction (assuming the model outputs probabilities for each class)
        prediction = model.predict(input_data_scaled)[0]  # Get the first prediction
        
        if isinstance(prediction, np.ndarray) and prediction.size == 1:
            prediction = float(prediction[0])
        elif isinstance(prediction, np.ndarray):
            prediction = prediction[0]  # Assuming binary classification, take the first element
        
        # Interpret the prediction
        if prediction > 0.5:
            result = 'The tumor characteristics suggest it may be Malignant'
            confidence = prediction * 100
        else:
            result = 'The tumor characteristics suggest it may be Benign'
            confidence = (1 - prediction) * 100
        
        return result, confidence
    except Exception as e:
        return str(e), 0


# Test cases
test_cases = {
    "Likely Benign": {
        'tumor_size': 8.0, 'tumor_texture': 12.0, 'tumor_smoothness': 0.08,
        'tumor_density': 0.05, 'tumor_irregularity': 0.02, 'tumor_symmetry': 0.15,
        'fractal_dimension': 0.05, 'radius_error': 0.2, 'texture_error': 0.3,
        'perimeter_error': 1.0, 'area_error': 10.0, 'smoothness_error': 0.002,
        'compactness_error': 0.005, 'concavity_error': 0.002, 'concave_points_error': 0.001,
        'symmetry_error': 0.005, 'fractal_dimension_error': 0.001
    },
    "Likely Malignant": {
        'tumor_size': 20.0, 'tumor_texture': 25.0, 'tumor_smoothness': 0.15,
        'tumor_density': 0.30, 'tumor_irregularity': 0.35, 'tumor_symmetry': 0.30,
        'fractal_dimension': 0.08, 'radius_error': 1.0, 'texture_error': 1.5,
        'perimeter_error': 5.0, 'area_error': 100.0, 'smoothness_error': 0.01,
        'compactness_error': 0.03, 'concavity_error': 0.03, 'concave_points_error': 0.01,
        'symmetry_error': 0.03, 'fractal_dimension_error': 0.008
    },
    "Borderline Case": {
        'tumor_size': 15.0, 'tumor_texture': 18.0, 'tumor_smoothness': 0.11,
        'tumor_density': 0.15, 'tumor_irregularity': 0.15, 'tumor_symmetry': 0.22,
        'fractal_dimension': 0.06, 'radius_error': 0.5, 'texture_error': 0.7,
        'perimeter_error': 2.5, 'area_error': 50.0, 'smoothness_error': 0.005,
        'compactness_error': 0.015, 'concavity_error': 0.015, 'concave_points_error': 0.005,
        'symmetry_error': 0.015, 'fractal_dimension_error': 0.004
    }
}
from flask_cors import CORS
# Streamlit app
st.title('Breast Cancer Risk Assessment Tool')




app = st.legacy_create_server()
CORS(app)


st.write("""
This tool uses detailed tumor characteristics to assess the potential risk of breast cancer. 
Please note that this is not a diagnostic tool and should not replace professional medical advice.
""")

# Add a selectbox for choosing between manual input and test cases
input_method = st.selectbox("Choose input method", ["Manual Input", "Test Cases"])

if input_method == "Manual Input":
    # User input for all 30 features
    tumor_size = st.slider('Tumor Size (mean radius)', 5.0, 30.0, 15.0, 0.1)
    tumor_texture = st.slider('Tumor Texture', 9.0, 40.0, 20.0, 0.1)
    tumor_smoothness = st.slider('Tumor Smoothness', 0.05, 0.20, 0.10, 0.01)
    tumor_density = st.slider('Tumor Density (compactness)', 0.02, 0.50, 0.10, 0.01)
    tumor_irregularity = st.slider('Tumor Irregularity (concavity)', 0.0, 0.50, 0.10, 0.01)
    tumor_symmetry = st.slider('Tumor Symmetry', 0.10, 0.40, 0.20, 0.01)
    fractal_dimension = st.slider('Fractal Dimension', 0.04, 0.10, 0.06, 0.001)

    st.subheader('Error Measurements')
    radius_error = st.slider('Radius Error', 0.1, 2.0, 0.5, 0.1)
    texture_error = st.slider('Texture Error', 0.1, 2.0, 0.5, 0.1)
    perimeter_error = st.slider('Perimeter Error', 0.5, 10.0, 2.0, 0.1)
    area_error = st.slider('Area Error', 10.0, 200.0, 50.0, 1.0)
    smoothness_error = st.slider('Smoothness Error', 0.001, 0.02, 0.005, 0.001)
    compactness_error = st.slider('Compactness Error', 0.002, 0.05, 0.01, 0.001)
    concavity_error = st.slider('Concavity Error', 0.002, 0.05, 0.01, 0.001)
    concave_points_error = st.slider('Concave Points Error', 0.001, 0.02, 0.005, 0.001)
    symmetry_error = st.slider('Symmetry Error', 0.002, 0.05, 0.01, 0.001)
    fractal_dimension_error = st.slider('Fractal Dimension Error', 0.001, 0.02, 0.005, 0.001)

    user_answers = {
        'tumor_size': tumor_size,
        'tumor_texture': tumor_texture,
        'tumor_smoothness': tumor_smoothness,
        'tumor_density': tumor_density,
        'tumor_irregularity': tumor_irregularity,
        'tumor_symmetry': tumor_symmetry,
        'fractal_dimension': fractal_dimension,
        'radius_error': radius_error,
        'texture_error': texture_error,
        'perimeter_error': perimeter_error,
        'area_error': area_error,
        'smoothness_error': smoothness_error,
        'compactness_error': compactness_error,
        'concavity_error': concavity_error,
        'concave_points_error': concave_points_error,
        'symmetry_error': symmetry_error,
        'fractal_dimension_error': fractal_dimension_error
    }

    if st.button('Predict'):
        result, confidence = predict_tumor(user_answers)
        st.subheader(f'Prediction Result: {result}')
        st.write(f'Confidence: {confidence:.2f}%')

elif input_method == "Test Cases":
    selected_case = st.selectbox("Choose a test case", list(test_cases.keys()))
    st.write(f"Selected test case: {selected_case}")

    if st.button('Predict Test Case'):
        result, confidence = predict_tumor(test_cases[selected_case])
        st.subheader(f'Prediction Result: {result}')
        st.write(f'Confidence: {confidence:.2f}%')
