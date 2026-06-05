import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from prediction import BengaliWordPredictor
from segmentation import BanglaWordSegmenter

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Bangla Character / Word Recognition",
    page_icon="✍️",
    layout="centered",
)

# =========================================================
# LOAD MODEL + LABELS (cached across reruns)
# =========================================================
@st.cache_resource
def get_predictor():
    return BengaliWordPredictor()

@st.cache_resource
def get_segmenter():
    return BanglaWordSegmenter()

predictor = get_predictor()
segmenter = get_segmenter()

# =========================================================
# SESSION STATE – initialise once
# =========================================================
if "results" not in st.session_state:
    st.session_state.results = None

# =========================================================
# STREAMLIT UI
# =========================================================
st.title("✍️ Bangla Handwritten Recognition")
st.caption("Draw inside the canvas below to predict Bangla text.")

# Mode selection
mode = st.radio(
    "Select recognition mode:",
    ["🔤 Single Character", "📝 Word (multi-character)"],
    horizontal=True,
    label_visibility="collapsed"
)

is_word_mode = mode == "📝 Word (multi-character)"

# Clear old results when mode is switched
if st.session_state.results is not None:
    stored_mode = st.session_state.results.get("mode")
    if (stored_mode == "single") != (not is_word_mode):
        st.session_state.results = None

# Canvas dynamic scaling and instructions
if is_word_mode:
    st.info("💡 Draw a full Bangla **word**. The app will segment it into individual characters and predict each one.")
    canvas_width, canvas_height, stroke_width = 600, 220, 8
else:
    st.info("💡 Draw a single Bangla **character**.")
    canvas_width, canvas_height, stroke_width = 300, 300, 10


# =========================================================
# CANVAS CENTERED
# =========================================================
_, canvas_col, _ = st.columns([1, 4, 1]) if not is_word_mode else (None, st.container(), None)

if not is_word_mode:
    with canvas_col:
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=stroke_width,
            stroke_color="white",
            background_color="black",
            height=canvas_height,
            width=canvas_width,
            drawing_mode="freedraw",
            key="canvas",
        )
else:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=stroke_width,
        stroke_color="white",
        background_color="black",
        height=canvas_height,
        width=canvas_width,
        drawing_mode="freedraw",
        key="canvas",
    )

# =========================================================
# ACTIONS
# =========================================================
st.write("") # Spacer
col_btn, col_clear = st.columns([3, 1])
with col_btn:
    predict_clicked = st.button("🔍 Predict Text", use_container_width=True, type="primary")
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.results = None
        st.rerun()

if predict_clicked:
    if canvas_result.image_data is None:
        st.warning("Please draw something on the canvas first!")
        st.stop()

    raw_img = Image.fromarray(canvas_result.image_data.astype("uint8"))

    # ── Single character ──────────────────────────────────
    if not is_word_mode:
        with st.spinner("Predicting…"):
            label, confidence = predictor.predict(raw_img)
        st.session_state.results = {
            "mode": "single",
            "label": label,
            "confidence": confidence,
        }

    # ── Word mode ─────────────────────────────────────────
    else:
        with st.spinner("Segmenting and predicting…"):
            char_images = segmenter.segment(raw_img)

        if not char_images:
            st.warning("No characters were detected. Please draw a word on the canvas.")
            st.session_state.results = None
            st.stop()

        characters = []
        with st.spinner(f"Classifying {len(char_images)} character(s)…"):
            for char_img in char_images:
                lbl, conf = predictor.predict(char_img)
                characters.append({"crop": char_img, "label": lbl, "confidence": conf})

        predicted_word = "".join(c["label"] for c in characters)
        avg_confidence = sum(c["confidence"] for c in characters) / len(characters)

        st.session_state.results = {
            "mode": "word",
            "characters": characters,
            "word": predicted_word,
            "avg_confidence": avg_confidence,
        }

# =========================================================
# DISPLAY RESULTS
# =========================================================
if st.session_state.results is not None:
    r = st.session_state.results
    st.markdown("---")

    # ── Single character result ───────────────────────────
    if r["mode"] == "single":
        st.subheader("Prediction Result")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric(label="Predicted Character", value=r['label'])
        m_col2.metric(label="Confidence", value=f"{r['confidence'] * 100:.2f}%")

    # ── Word result ───────────────────────────────────────
    else:
        st.subheader("Word Prediction")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric(label="Predicted Word", value=r['word'])
        m_col2.metric(label="Average Confidence", value=f"{r['avg_confidence'] * 100:.2f}%")

        st.write("")
        st.subheader(f"Character Breakdown ({len(r['characters'])} detected)")
        
        # Clean responsive grid layout for dynamic character lengths
        # Maximum 5 columns per row to prevent layout breakdown on small layouts
        max_cols = min(len(r["characters"]), 5)
        cols = st.columns(max_cols)
        
        for idx, ch in enumerate(r["characters"]):
            col_target = cols[idx % max_cols]
            with col_target:
                st.image(ch["crop"], use_container_width=True)
                st.markdown(
                    f"<div style='text-align:center; font-size:1.8rem; font-weight:bold; line-height:1.2; margin-top:2px;'>"
                    f"{ch['label']}</div>"
                    f"<div style='text-align:center; color:#888; font-size:0.8rem; margin-bottom:15px;'>"
                    f"{ch['confidence']*100:.1f}%</div>",
                    unsafe_allow_html=True,
                )