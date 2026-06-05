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
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    /* ── Global ──────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .block-container {
        padding-top: 2rem;
        max-width: 720px;
    }

    /* ── Header ──────────────────────────────────────────── */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .app-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #4fc3f7, #ab47bc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-header p {
        color: #999;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }

    /* ── Card wrapper ────────────────────────────────────── */
    .card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        background: rgba(255,255,255,0.02);
    }
    .card-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888;
        margin-bottom: 0.75rem;
    }

    /* ── Result cards ────────────────────────────────────── */
    .result-banner {
        text-align: center;
        padding: 1.5rem 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(79,195,247,0.08), rgba(171,71,188,0.08));
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 1rem;
    }
    .result-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #999;
        margin-bottom: 0.25rem;
    }
    .result-value {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #fff;
        line-height: 1.2;
    }
    .result-confidence {
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    .conf-high { color: #66bb6a; }
    .conf-mid  { color: #ffa726; }
    .conf-low  { color: #ef5350; }

    /* ── Character grid cards ────────────────────────────── */
    .char-card {
        text-align: center;
        padding: 0.6rem 0.4rem;
        border-radius: 10px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
    }
    .char-label {
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.2;
        margin-top: 4px;
    }
    .char-conf {
        font-size: 0.75rem;
        margin-bottom: 4px;
    }

    /* ── Divider ─────────────────────────────────────────── */
    .soft-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 1.25rem 0;
    }

    /* ── Hide default streamlit title ────────────────────── */
    .stApp header { visibility: hidden; }

    /* ── Mode radio style tweaks ─────────────────────────── */
    div[data-testid="stRadio"] > div {
        gap: 0.5rem;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

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
# HEADER
# =========================================================
st.markdown("""
<div class="app-header">
    <h1>✍️ Bangla Handwritten Recognition</h1>
    <p>Draw on the canvas and hit <strong>Predict</strong> to recognize Bangla text</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# MODE SELECTION
# =========================================================
st.markdown('<div class="card-title" style="text-align:center;">Recognition Mode</div>', unsafe_allow_html=True)
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

st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

# =========================================================
# CANVAS
# =========================================================
if is_word_mode:
    canvas_width, canvas_height, stroke_width = 600, 220, 8
    hint = "Draw a full Bangla <strong>word</strong> below"
else:
    canvas_width, canvas_height, stroke_width = 300, 300, 10
    hint = "Draw a single Bangla <strong>character</strong> below"

st.markdown(f'<div class="card-title" style="text-align:center;">{hint}</div>', unsafe_allow_html=True)

# Centre the canvas for single-character mode
if not is_word_mode:
    _, canvas_col, _ = st.columns([1, 4, 1])
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
# PREDICT BUTTON
# =========================================================
st.write("")
predict_clicked = st.button("🔍 Predict", use_container_width=True, type="primary")

# =========================================================
# PREDICTION LOGIC
# =========================================================
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
def _conf_class(c: float) -> str:
    """Return CSS class based on confidence value."""
    if c >= 0.80:
        return "conf-high"
    if c >= 0.50:
        return "conf-mid"
    return "conf-low"

if st.session_state.results is not None:
    r = st.session_state.results
    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # ── Single character result ───────────────────────────
    if r["mode"] == "single":
        conf = r["confidence"]
        cls = _conf_class(conf)
        st.markdown(f"""
        <div class="result-banner">
            <div class="result-label">Predicted Character</div>
            <div class="result-value">{r['label']}</div>
            <div class="result-confidence {cls}">{conf * 100:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Word result ───────────────────────────────────────
    else:
        avg = r["avg_confidence"]
        cls = _conf_class(avg)
        st.markdown(f"""
        <div class="result-banner">
            <div class="result-label">Predicted Word</div>
            <div class="result-value">{r['word']}</div>
            <div class="result-confidence {cls}">{avg * 100:.1f}% avg confidence</div>
        </div>
        """, unsafe_allow_html=True)

        # Character breakdown
        st.markdown(
            f'<div class="card-title" style="text-align:center; margin-top:0.5rem;">'
            f'Character Breakdown — {len(r["characters"])} detected</div>',
            unsafe_allow_html=True,
        )

        max_cols = min(len(r["characters"]), 5)
        cols = st.columns(max_cols)

        for idx, ch in enumerate(r["characters"]):
            col_target = cols[idx % max_cols]
            conf_cls = _conf_class(ch["confidence"])
            with col_target:
                st.markdown('<div class="char-card">', unsafe_allow_html=True)
                st.image(ch["crop"], use_container_width=True)
                st.markdown(
                    f'<div class="char-label">{ch["label"]}</div>'
                    f'<div class="char-conf {conf_cls}">{ch["confidence"]*100:.1f}%</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)