# app.py  -- DR Ensemble Model Inference App

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = "ensemble_effb3.keras"
IMG_SIZE = 256
CLASS_NAMES = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]


@st.cache_resource
def load_model():
    st.write("Loading model from disk... (first time only)")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image).astype("float32")
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


st.set_page_config(page_title="Diabetic Retinopathy Detection", page_icon="🧿", layout="centered")

st.title("🧿 Diabetic Retinopathy Detection (Ensemble EfficientNetB3)")
st.write(
    """
Upload a **retinal fundus image** and the app will predict the **DR stage**  
using your **EfficientNetB3 Ensemble model** (cross-entropy + focal loss).
"""
)

with st.sidebar:
    st.header("Model Details")
    st.markdown(
        """
- Backbone: **EfficientNetB3**
- Technique: Class weights + Focal loss + Ensemble
- Test Accuracy (Ensemble): **~66%**
- Classes:
  - 0: No DR  
  - 1: Mild  
  - 2: Moderate  
  - 3: Severe  
  - 4: Proliferative
        """
    )

uploaded_file = st.file_uploader("Upload retinal image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Fundus Image", use_column_width=True)

    if st.button("🔍 Predict DR Stage"):
        with st.spinner("Running inference on ensemble model..."):
            model = load_model()
            x = preprocess_image(image)
            logits = model.predict(x)[0]
            probs = tf.nn.softmax(logits).numpy()

            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]

        st.subheader("Prediction Result")
        st.write(f"**Predicted Class:** {pred_class}")

        st.subheader("Class Probabilities")
        st.table({
            "Class": CLASS_NAMES,
            "Probability (%)": [f"{p*100:.2f}" for p in probs]
        })

        st.info(
            "⚠️ This is an academic / research model, not a clinical diagnostic tool. "
            "Predictions must be validated by an ophthalmologist."
        )
else:
    st.info("Please upload a **fundus image** to start.")