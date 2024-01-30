import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

# Load your trained model
model = load_model(r'C:\Users\Dell 5470\Desktop\Aadhaar Detection Using Streamlit\model\Aasteroid.h5')

def predict_aadhaar(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Make predictions
    prediction = model.predict(img_array)

    # Assuming your model has two classes (0: Not Aadhaar, 1: Aadhaar)
    predicted_class = np.argmax(prediction)

    # Return True if predicted as Aadhaar, False otherwise
    return predicted_class == 1

def main():
    st.title("Aadhaar Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_path = "temp_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        if predict_aadhaar(image_path):
            st.success("This is an Aadhaar card!")
        else:
            st.warning("This is not an Aadhaar card.")

if __name__ == "__main__":
    main()
