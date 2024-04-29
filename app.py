import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO

# Load the pre-trained model
model = load_model("cifar10_classification_model.h5")

# Define the classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to preprocess the image
def preprocess_image(image_data):
    img = image.load_img(image_data, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    return classes[predicted_class], confidence

# Streamlit app
def main():
    st.title("CIFAR-10 Image Classifier")
    st.write("Upload an image for classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Classify'):
            prediction, confidence = predict(uploaded_file)
            st.write(f"Prediction: {prediction}, Confidence: {confidence}")

if __name__ == '__main__':
    main()
