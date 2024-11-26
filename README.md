# Plant Disease Classifier - Streamlit App

This is a simple web application built using Streamlit to classify plant diseases from images using a pre-trained deep learning model. The app can identify various plant diseases, including those affecting apple, grape, tomato, and more.

## Features

- Upload an image of a plant leaf affected by a disease.
- Classify the disease from the uploaded image using a pre-trained deep learning model.
- Display the classification result with the predicted disease.

## Requirements

Before running the app, make sure you have the following Python libraries installed:

- `streamlit`
- `tensorflow`
- `pillow`
- `numpy`

You can install these dependencies using `pip`:

```bash
pip install streamlit tensorflow pillow numpy
```

## Files

- **plant_disease_prediction_model.h5**: The pre-trained model for plant disease classification.
- **class_indices.json**: A JSON file that maps class indices to disease names.
- **app.py**: The main Streamlit app code.

## How to Use

1. Clone or download the repository.
2. Ensure that the model file (`plant_disease_prediction_model.h5`) and class indices file (`class_indices.json`) are in the `trained_model/` directory.
3. Run the app with the following command:

```bash
streamlit run app.py
```

4. On the web interface, upload an image of a plant leaf.
5. Click the "Classify" button to predict the plant disease.
6. The app will display the predicted disease name based on the uploaded image.

## Image Classification Process

The app uses the following steps to classify an uploaded image:

1. **Image Preprocessing**:

   - The image is resized to 224x224 pixels.
   - The image is converted to a NumPy array and scaled to values between 0 and 1.
   - A batch dimension is added to make it compatible with the model's input shape.

2. **Prediction**:
   - The preprocessed image is passed through the model.
   - The model outputs the predicted class index, which is mapped to a disease name using the `class_indices.json` file.

## Supported Classes

The model can classify the following plant diseases:

- Apple (Apple scab, Black rot, Cedar apple rust, healthy)
- Blueberry (healthy)
- Cherry (Powdery mildew, healthy)
- Corn (Cercospora leaf spot, Common rust, Northern leaf blight, healthy)
- Grape (Black rot, Esca, Leaf blight, healthy)
- Orange (Citrus greening)
- Peach (Bacterial spot, healthy)
- Pepper (Bacterial spot, healthy)
- Potato (Early blight, Late blight, healthy)
- Raspberry (healthy)
- Soybean (healthy)
- Squash (Powdery mildew)
- Strawberry (Leaf scorch, healthy)
- Tomato (Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, healthy)

### Dataset used to train the Model -

Kaggle Dataset Link: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
