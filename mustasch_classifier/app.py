import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(
    page_title="Mustache Power Rating™",
    page_icon="🥸",
    layout="centered"
)

# --------------------------------------------------
# Load models
# --------------------------------------------------

@st.cache_resource
def load_models():
    mustache_model = tf.keras.models.load_model(
        "models/mustache_detector.keras"
    )

    epic_model = tf.keras.models.load_model(
        "models/epic_detector.keras"
    )

    return mustache_model, epic_model


mustache_model, epic_model = load_models()

# --------------------------------------------------
# Constants
# --------------------------------------------------

IMG_SIZE = (128, 128)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.image("assets/logo.png", width=320)

st.title("Mustache Power Rating™")

st.caption(
    "Official Facial Hair Assessment "
    "by the International Mustache Authority"
)

st.write(
    "Upload a face. Our scientists will determine "
    "whether a mustache exists and, if so, "
    "its official power rating."
)

# --------------------------------------------------
# Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Submit specimen for analysis",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def prepare_image(image):

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    arr = np.array(image)
    arr = np.expand_dims(arr, axis=0)

    return arr


def classify_epicness(score):

    if score >= 0.85:
        return (
            "🏆 LEGENDARY STACHE",
            "The International Mustache Authority is impressed.",
            "assets/epic.mp4"
        )

    elif score >= 0.65:
        return (
            "🎩 Distinguished Gentleman",
            "A respectable upper-lip performance.",
            "assets/medium.mp4"
        )

    elif score >= 0.45:
        return (
            "🧔 Standard Issue Mustache",
            "Acceptable. Not historic.",
            "assets/medium.mp4"
        )

    elif score >= 0.25:
        return (
            "🌱 Apprentice Stache",
            "Mustache growth is currently in beta testing.",
            "assets/not_epic.mp4"
        )

    else:
        return (
            "🪶 Fjunig Mustache",
            "The mustache exists mostly as a theoretical concept.",
            "assets/not_epic.mp4"
        )

# --------------------------------------------------
# Analysis
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Submitted specimen",
        use_container_width=True
    )

    img_array = prepare_image(image)

    st.write("mustache_prob:", mustache_model.predict(img_array, verbose=0)[0][0])

    # DEBUG: visa vad modellen faktiskt får in
    st.image(
        img_array[0].astype("uint8"),
        caption="Vad modellen ser",
        width=200
    )

    # --------------------------
    # Mustache detector
    # --------------------------

    mustache_prob = float(
        mustache_model.predict(
            img_array,
            verbose=0
        )[0][0]
    )

    st.subheader("Facial Hair Report")

    st.write(
        f"Mustache Detection Confidence: "
        f"**{mustache_prob * 100:.1f}%**"
    )

    if mustache_prob < 0.5:

        st.error("NO CERTIFIED MUSTACHE DETECTED")

        st.write(
            "Verdict: Upper lip currently operating "
            "without sufficient authority."
        )

    else:

        st.success(
            "Mustache detected. "
            "Initiating epicness protocol."
        )

        # --------------------------
        # Epic detector
        # --------------------------

        thin_prob = float(
            epic_model.predict(
                img_array,
                verbose=0
            )[0][0]
        )

        epic_score = 1 - thin_prob

        st.info("DEBUG ACTIVE")
        st.write("mustache_prob:", mustache_prob)
        st.write("thin_prob:", thin_prob)
        st.write("epic_score:", epic_score)

        title, description, video_file = classify_epicness(
            epic_score
        )

        st.metric(
            "Mustache Power Rating™",
            f"{epic_score * 100:.1f}/100"
        )

        st.progress(float(epic_score))

        st.subheader(title)

        st.write(description)

        if epic_score >= 0.85:
            st.balloons()

        if video_file:

            if os.path.exists(video_file):
                st.video(video_file)
            else:
                st.warning(
                    f"Video file not found: {video_file}"
                )

        st.divider()

        st.caption(
            "Results certified by the "
            "International Mustache Authority™"
        )