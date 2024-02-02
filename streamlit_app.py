import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model_path = r'C:\Users\Dell 5470\OneDrive\Desktop\Aadhaar Detection Using Streamlit\model\Ashes.h5'
model = load_model(model_path)

def predict_aadhaar(image_path):
    try:
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
    except Exception as e:
        st.error(f"Error predicting the image: {e}")
        return False

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
