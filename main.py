import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

import openai

OPENAI_KEY = st.secrets["OPENAI_API_KEY"]

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title('Plant Disease Classifier 🌱')

# Allow user to either upload an image or take a picture
picture = st.camera_input("Take a picture")
uploaded_image = st.file_uploader("Or upload an image...", type=["jpg", "jpeg", "png"])

# Handle if either image is uploaded or camera input is taken
if uploaded_image is not None:
    image = Image.open(uploaded_image)
elif picture is not None:
    image = Image.open(picture)
else:
    image = None


# If an image is provided, display the image and classify it
if image is not None:
    col1, col2 = st.columns(2)

    resized_img = image.resize((150, 150))
    st.image(resized_img)

    if st.button('Classify'):
        # Preprocess the uploaded image and predict the class
        prediction = predict_image_class(model, uploaded_image if uploaded_image is not None else picture, class_indices)
        
        st.success(f'Prediction: {str(prediction)}')

        # Request disease information from OpenAI API
        openai.api_key = OPENAI_KEY
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
                You are a plant disease expert. When provided with the name of a plant disease, you will respond with the following information:

                1. A brief description of the disease, including the plant it affects and its symptoms.
                2. The cause of the disease (e.g., fungus, bacteria, virus).
                3. Potential methods for controlling or curing the disease, including chemical, organic, and preventive treatments.
                4. Any environmental conditions or practices that may help prevent the disease in the future.

                If the disease name is 'healthy' or indicates a healthy plant, your response should be:
                - The plant is healthy, with no visible signs of disease or pests. Continue to provide good care, such as regular watering, sunlight, and proper soil management to maintain plant health.

                The name of the disease will be provided as input, and you should respond accordingly.
                """},
                {"role": "user", "content": f"The disease is {str(prediction)}"},
            ],
        )

        openai_response = response.choices[0].message.content
        st.subheader("Disease Information")
        st.write(openai_response)