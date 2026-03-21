# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image  # ← fix 1: use tensorflow.keras not keras
from PIL import Image

st.set_page_config(page_title="Dogs vs Cats Classifier", page_icon="🐾")
st.title("🐾 Dogs vs Cats Classifier")
st.caption("Powered by MobileNetV2 — 97% accuracy")

@st.cache_resource
def load():
    return load_model('best_model.keras')  # ← fix 2: use your actual saved model name

model = load()

CLASS_NAMES = ['Cat', 'Dog']  # ← fix 3: replace with your actual class names (alphabetical order)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert('RGB')  # ← fix 4: force RGB, some images are RGBA/grayscale
    img_resized = img.resize((128, 128))
    st.image(img, caption="Uploaded image", width=300)  # show original size

    arr = image.img_to_array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0][0]
    label = CLASS_NAMES[1] if pred > 0.5 else CLASS_NAMES[0]
    confidence = float(pred if pred > 0.5 else 1 - pred)  # ← fix 5: cast to float

    st.success(f"Prediction  : **{label}**")
    st.info(f"Confidence  : **{confidence * 100:.2f}%**")
    st.progress(confidence)  # ← bonus: visual confidence bar