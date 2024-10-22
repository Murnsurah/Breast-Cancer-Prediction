import pickle 
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Define feature names
columns = [
    'radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
    'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'smoothness_se', 'compactness_se',
    'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

def main():
    st.title("Breast Cancer Prediction")
    
    # Custom HTML template for header
    html_temp = """
    <div style="background-color:#025246;padding:10px">
    <h2 style="color:white;text-align:center;">Breast Cancer Prediction App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Load and display image
    image = Image.open('Breast Cancer.jpg')
    st.image(image, caption='', use_column_width=True)

    # Create input fields for all features
    radius_mean = st.number_input("radius_mean", min_value=0.0, value=0.0)
    texture_mean = st.number_input("texture_mean", min_value=0.0, value=0.0)
    smoothness_mean = st.number_input("smoothness_mean", min_value=0.0, value=0.0)
    compactness_mean = st.number_input("compactness_mean", min_value=0.0, value=0.0)
    concavity_mean = st.number_input("concavity_mean", min_value=0.0, value=0.0)
    symmetry_mean = st.number_input("symmetry_mean", min_value=0.0, value=0.0)
    fractal_dimension_mean = st.number_input("fractal_dimension_mean", min_value=0.0, value=0.0)
    radius_se = st.number_input("radius_se", min_value=0.0, value=0.0)
    texture_se = st.number_input("texture_se", min_value=0.0, value=0.0)
    smoothness_se = st.number_input("smoothness_se", min_value=0.0, value=0.0)
    compactness_se = st.number_input("compactness_se", min_value=0.0, value=0.0)
    concavity_se = st.number_input("concavity_se", min_value=0.0, value=0.0)
    concave_points_se = st.number_input("concave points_se", min_value=0.0, value=0.0)
    symmetry_se = st.number_input("symmetry_se", min_value=0.0, value=0.0)
    fractal_dimension_se = st.number_input("fractal_dimension_se", min_value=0.0, value=0.0)
    smoothness_worst = st.number_input("smoothness_worst", min_value=0.0, value=0.0)
    compactness_worst = st.number_input("compactness_worst", min_value=0.0, value=0.0)
    concavity_worst = st.number_input("concavity_worst", min_value=0.0, value=0.0)
    symmetry_worst = st.number_input("symmetry_worst", min_value=0.0, value=0.0)
    fractal_dimension_worst = st.number_input("fractal_dimension_worst", min_value=0.0, value=0.0)

    # Prediction button
    if st.button("Predict"): 
        # Collect all feature values
        features = [[radius_mean, texture_mean, smoothness_mean, compactness_mean,
                     concavity_mean, symmetry_mean, fractal_dimension_mean,
                     radius_se, texture_se, smoothness_se, compactness_se,
                     concavity_se, concave_points_se, symmetry_se,
                     fractal_dimension_se, smoothness_worst, compactness_worst,
                     concavity_worst, symmetry_worst, fractal_dimension_worst]]
        
        # Make a prediction
        prediction = loaded_model.predict(features)
        output = int(prediction[0])
        
        # Map the prediction to a result
        if output == 1:
            st.success('The prediction is: Malignant (Cancerous)')
        else:
            st.success('The prediction is: Benign (Non-Cancerous)')

if __name__ == '__main__':
    main()
