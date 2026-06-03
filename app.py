import streamlit as st
import numpy as np
import tensorflow as tf
import json
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "models/model.keras"
LABELS_JSON_PATH = "labels.json"
IMAGE_SIZE = 224

# =========================================================
# LOAD MODEL + LABELS
# =========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

with open(LABELS_JSON_PATH, "r", encoding="utf-8") as f:
    label_mapping = json.load(f)

sorted_class_ids = sorted(label_mapping.keys(), key=int)
index_to_label = {idx: label_mapping[cid] for idx, cid in enumerate(sorted_class_ids)}

# =========================================================
# PREPROCESS FUNCTION
# =========================================================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =========================================================
# STREAMLIT UI
# =========================================================
st.title("✍️ Bangla Character Recognition App")
st.write("Draw a Bangla character below and click **Predict**.")

# Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("🔍 Predict"):

    if canvas_result.image_data is None:
        st.warning("Please draw something first!")
        st.stop()

    # Convert canvas to image
    img = Image.fromarray(canvas_result.image_data.astype("uint8"))

    # Preprocess
    input_tensor = preprocess_image(img)

    # Predict
    predictions = model.predict(input_tensor, verbose=0)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    predicted_label = index_to_label[predicted_index]

    # Output
    st.subheader("Prediction Result")
    st.success(f"🧠 Predicted Character: **{predicted_label}**")
    st.info(f"📊 Confidence: **{confidence * 100:.2f}%**")