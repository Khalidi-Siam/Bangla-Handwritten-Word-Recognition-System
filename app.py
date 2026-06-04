import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from prediction import BengaliWordPredictor

# =========================================================
# LOAD MODEL + LABELS
# =========================================================
@st.cache_resource
def get_predictor():
    return BengaliWordPredictor()

predictor = get_predictor()

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

    # Predict
    predicted_label, confidence = predictor.predict(img)

    # Output
    st.subheader("Prediction Result")
    st.success(f"🧠 Predicted Character: **{predicted_label}**")
    st.info(f"📊 Confidence: **{confidence * 100:.2f}%**")