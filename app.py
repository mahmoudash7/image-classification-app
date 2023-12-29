import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
import os

st.title("Image Classification Web App")

# Specify the absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'img_svc_model.p')

# Check if the model file exists
if os.path.exists(model_path):
    # Load the trained model
    model = pickle.load(open(model_path, 'rb'))
else:
    st.error("Error: Model file not found.")

CATEGORIES = ['modern bicycle', 'violin', 'kangaroo']

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the uploaded image
    img = imread(uploaded_file)
    img_resized = resize(img, (150, 150, 3))
    flat_data_test = np.array([img_resized.flatten()])

    # Make predictions
    y_out = model.predict(flat_data_test)
    category = CATEGORIES[y_out[0]]

    # Display the image and prediction
    st.image(img, caption=f'Predicted Output: {category}', use_column_width=True)
