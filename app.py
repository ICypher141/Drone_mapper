import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st

# Load the trained model
model = load_model('simple_drone_segmentation_model.h5')

# Define constants
INPUT_SHAPE = (256, 256)  # Image size for the model
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Streamlit App Title
st.title("🌍 Drone Image Segmentation App")
st.markdown(
    """
    **Upload an aerial drone image**, and this app will generate a segmentation mask
    predicting areas of interest using a trained U-Net model.
    """
)

# Sidebar for user instructions
st.sidebar.header("How to use:")
st.sidebar.markdown(
    """
    1. Upload an image in JPG/PNG format.
    2. Wait for the app to process and generate the mask.
    3. View and download the predicted segmentation mask.
    """
)

# File uploader widget
uploaded_file = st.file_uploader("Upload an aerial image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Save uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        image_path = os.path.join(UPLOAD_FOLDER, "input_image.jpg")
        image.save(image_path)

        # Display the input image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for the model
        resized_image = image.resize(INPUT_SHAPE)
        image_array = np.array(resized_image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Perform prediction
        st.text("Processing the image...")
        predictions = model.predict(image_array)

        # Generate the mask
        predicted_mask = np.argmax(predictions[0], axis=-1).astype(np.uint8)
        mask_image = Image.fromarray((predicted_mask * (255 // predicted_mask.max())).astype(np.uint8))
        mask_path = os.path.join(UPLOAD_FOLDER, "predicted_mask.png")
        mask_image.save(mask_path)

        # Display the segmentation result
        st.image(mask_image, caption="Predicted Segmentation Mask", use_column_width=True)

        # Option to download the mask
        with open(mask_path, "rb") as file:
            st.download_button(
                label="📥 Download Segmentation Mask",
                data=file,
                file_name="predicted_mask.png",
                mime="image/png"
            )

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
